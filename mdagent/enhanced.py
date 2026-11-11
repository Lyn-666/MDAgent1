"""Enhanced sampling of side-chain chi1 angles with well-tempered metadynamics.

Plain MD cannot tell whether a side chain that never flips is trapped, or
has only one stable state. Metadynamics answers this. It adds a bias
potential along chi1 that grows wherever the side chain has been, so the
side chain is pushed over the barriers. In the well-tempered form, the bias
grows more slowly where it is already large. At the end, the free energy
along chi1 is F(chi) = -(gamma / (gamma - 1)) * V(chi), where V is the bias
and gamma is the bias factor.

All residues are biased in one simulation. Each chi1 has its own
one-dimensional bias. All biases live in one OpenMM CustomCVForce, so the
cost is small.

Each replica starts from the final state of a plain MD replica. It runs in
chunks. After each chunk the state, the bias and the chi1 samples are saved,
so a stopped run can resume.

Limitation: each residue's free energy comes from its own bias only. If two
biased side chains are strongly coupled, the result for each one is less
accurate. The two-replica check and the first-half/second-half check catch
many such cases as "not_converged".
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import mdtraj as md
import numpy as np
import openmm

from .diagnose import STATE_CENTERS_DEG, STATE_NAMES, angular_distance_deg, assign_states, count_transitions
from .simulate import MDSettings, SystemFiles, make_simulation

KB_KJ = 0.0083144626  # Boltzmann constant in kJ/(mol K)
ResidueKey = tuple[str, int]


@dataclass
class MetadSettings:
    length_ns: float = 20.0       # Per replica.
    n_replicas: int = 2
    bias_factor: float = 10.0     # gamma. Larger pushes harder over barriers.
    height_kj: float = 1.2        # Initial height of each Gaussian.
    sigma_deg: float = 12.0       # Width of each Gaussian.
    pace_ps: float = 1.0          # Time between Gaussians.
    grid_points: int = 361        # Grid on chi1 from -180° to 180° (both ends included).
    chunk_ns: float = 1.0         # Save progress after each chunk.
    single_state_cutoff: float = 0.95   # Main state above this population: one stable state.
    agreement_tolerance: float = 0.10   # Max population difference between replicas and halves.
    min_crossings: int = 4               # Min chi1 state changes per replica under bias.


def chi1_quads(topology: openmm.app.Topology, residues: list[ResidueKey]) -> dict[ResidueKey, tuple]:
    """Atom indices (N, CA, CB, CG-like) of chi1 for each residue, in the full system."""
    top = md.Topology.from_openmm(topology)
    traj = md.Trajectory(np.zeros((1, top.n_atoms, 3)), top)
    idx, _ = md.compute_chi1(traj)
    out = {}
    for quad in idx:
        res = top.atom(int(quad[1])).residue
        key = (res.chain.chain_id, res.resSeq)
        if key in residues:
            out[key] = tuple(int(i) for i in quad)
    missing = [k for k in residues if k not in out]
    if missing:
        raise ValueError(f"No chi1 angle for residues {missing}.")
    return out


class ChiBiases:
    """One periodic well-tempered bias on chi1 for each residue, in one force."""

    def __init__(self, quads: list[tuple], s: MetadSettings, temperature_k: float,
                 bias: np.ndarray | None = None):
        self.s = s
        self.kt = KB_KJ * temperature_k
        self.grid = np.linspace(-math.pi, math.pi, s.grid_points)
        self.bias = np.zeros((len(quads), s.grid_points)) if bias is None else bias.copy()
        self.sigma = math.radians(s.sigma_deg)
        terms = []
        self.force = openmm.CustomCVForce("0")
        self.tables = []
        for i, quad in enumerate(quads):
            torsion = openmm.CustomTorsionForce("theta")
            torsion.addTorsion(*quad)
            self.force.addCollectiveVariable(f"chi{i}", torsion)
            table = openmm.Continuous1DFunction(self.bias[i].tolist(), -math.pi, math.pi, True)
            self.force.addTabulatedFunction(f"bias{i}", table)
            self.tables.append(table)
            terms.append(f"bias{i}(chi{i})")
        self.force.setEnergyFunction(" + ".join(terms))

    def values(self, context) -> np.ndarray:
        return np.array(self.force.getCollectiveVariableValues(context))

    def bias_at(self, i: int, x: float) -> float:
        return float(np.interp(x, self.grid, self.bias[i]))

    def deposit(self, x: np.ndarray) -> None:
        gamma = self.s.bias_factor
        for i, xi in enumerate(x):
            height = self.s.height_kj * math.exp(-self.bias_at(i, xi) / (self.kt * (gamma - 1)))
            d = (self.grid - xi + math.pi) % (2 * math.pi) - math.pi
            self.bias[i] += height * np.exp(-d * d / (2 * self.sigma ** 2))
        self.bias[:, -1] = self.bias[:, 0]  # Both ends are the same angle.

    def push(self, context) -> None:
        for i, table in enumerate(self.tables):
            table.setFunctionParameters(self.bias[i].tolist(), -math.pi, math.pi)
        self.force.updateParametersInContext(context)


def _paths(run_dir: Path, replica: int) -> dict[str, Path]:
    d = run_dir / "enhanced"
    return {
        "dir": d,
        "progress": d / f"replica_{replica}.json",
        "state": d / f"replica_{replica}.state.xml",
        "bias": d / f"replica_{replica}_bias.npy",
        "bias_half": d / f"replica_{replica}_bias_half.npy",
        "chi": d / f"replica_{replica}_chi.npy",
    }


def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def run_replica(
    files: SystemFiles,
    run_dir: Path,
    replica: int,
    start_state: Path,
    residues: list[ResidueKey],
    s: MetadSettings,
    md_settings: MDSettings,
    seed: int,
    log: Callable[[str], None] = print,
) -> None:
    p = _paths(run_dir, replica)
    p["dir"].mkdir(exist_ok=True)
    quads = chi1_quads(files.topology, residues)
    progress = json.loads(p["progress"].read_text()) if p["progress"].exists() else {"ns_done": 0.0}
    if progress["ns_done"] >= s.length_ns - 1e-9:
        log(f"enhanced replica {replica}: already complete ({progress['ns_done']:g} ns).")
        return

    resuming = progress["ns_done"] > 0
    bias = np.load(p["bias"]) if resuming else None
    chi = list(np.load(p["chi"])) if resuming else []
    biases = ChiBiases([quads[k] for k in residues], s, md_settings.temperature_k, bias)
    sim = make_simulation(files, md_settings, seed, extra_forces=[biases.force])
    sim.loadState(str(p["state"] if resuming else start_state))
    biases.push(sim.context)

    steps_per_ps = round(1000.0 / md_settings.timestep_fs)
    pace = round(s.pace_ps * steps_per_ps)
    t0 = time.time()
    ns_start = progress["ns_done"]
    while progress["ns_done"] < s.length_ns - 1e-9:
        chunk_ns = min(s.chunk_ns, s.length_ns - progress["ns_done"])
        n_pace = round(chunk_ns * 1000 / s.pace_ps)
        for _ in range(n_pace):
            sim.step(pace)
            x = biases.values(sim.context)
            chi.append(x)
            biases.deposit(x)
            biases.push(sim.context)
        before = progress["ns_done"]
        progress["ns_done"] = round(before + chunk_ns, 9)
        half = s.length_ns / 2
        if before < half <= progress["ns_done"]:
            _save_npy(p["bias_half"], biases.bias)
        tmp = p["state"].with_suffix(".tmp")
        sim.saveState(str(tmp))
        os.replace(tmp, p["state"])
        _save_npy(p["bias"], biases.bias)
        _save_npy(p["chi"], np.array(chi))
        p["progress"].write_text(json.dumps(progress))
        speed = (progress["ns_done"] - ns_start) / ((time.time() - t0) / 86400)
        log(f"enhanced replica {replica}: {progress['ns_done']:g} of {s.length_ns:g} ns "
            f"({speed:.0f} ns/day)")


# ---------------------------------------------------------------- analysis

def free_energy(bias: np.ndarray, s: MetadSettings, temperature_k: float) -> np.ndarray:
    """F(chi) in kJ/mol, with minimum 0, from a well-tempered bias."""
    gamma = s.bias_factor
    f = -gamma / (gamma - 1) * bias
    return f - f.min()


def populations(f: np.ndarray, grid: np.ndarray, temperature_k: float) -> dict[str, float]:
    """Population of g+, t and g-. Each grid point belongs to the nearest state center."""
    kt = KB_KJ * temperature_k
    deg = np.degrees(grid[:-1])  # The last point repeats the first.
    w = np.exp(-f[:-1] / kt)
    dist = np.stack([angular_distance_deg(deg, c) for c in STATE_CENTERS_DEG.values()])
    nearest = dist.argmin(axis=0)
    total = w.sum()
    return {name: float(w[nearest == i].sum() / total) for i, name in enumerate(STATE_NAMES)}


@dataclass
class EnhancedResult:
    chain: str
    resseq: int
    resname: str
    populations: list[dict[str, float]]      # Per replica, at the end.
    populations_half: list[dict[str, float]]  # Per replica, at half time.
    crossings: list[int]
    free_energy: list[list[float]] = field(repr=False, default_factory=list)  # Per replica, on the grid.
    verdict: str = ""
    reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.resname}{self.resseq}"

    @property
    def mean_populations(self) -> dict[str, float]:
        return {n: float(np.mean([p[n] for p in self.populations])) for n in STATE_NAMES}


def judge(r: EnhancedResult, s: MetadSettings, temperature_k: float) -> None:
    kt = KB_KJ * temperature_k
    mean = r.mean_populations
    between = max(abs(a[n] - b[n]) for n in STATE_NAMES
                  for i, a in enumerate(r.populations) for b in r.populations[i + 1:]) \
        if len(r.populations) > 1 else 0.0
    drift = max(abs(a[n] - b[n]) for n in STATE_NAMES for a, b in zip(r.populations, r.populations_half)) \
        if r.populations_half else 0.0
    pops = ", ".join(f"{n} {mean[n]:.0%}" for n in STATE_NAMES)

    if min(r.crossings) < s.min_crossings:
        r.verdict = "not_converged"
        r.reason = (f"Even with the bias, chi1 changed state only {r.crossings} times per replica. "
                    "Run the enhanced sampling longer.")
    elif between > s.agreement_tolerance or drift > s.agreement_tolerance:
        r.verdict = "not_converged"
        r.reason = (f"Populations still change: replicas differ by up to {between:.0%}, and the "
                    f"populations changed by up to {drift:.0%} in the second half. Current estimate: {pops}.")
    else:
        main = max(mean, key=mean.get)
        others = [n for n in STATE_NAMES if n != main]
        gaps = {n: -kt * math.log(max(mean[n], 1e-12) / mean[main]) for n in others}
        if mean[main] >= s.single_state_cutoff:
            r.verdict = "single_state"
            r.reason = (f"chi1 has one stable state: {main} ({mean[main]:.0%}). The other states are "
                        + " and ".join(f"{n} {gaps[n]:.1f} kJ/mol" for n in others)
                        + " higher. Plain MD was not trapped; this side chain is sampled.")
        else:
            r.verdict = "multiple_states"
            r.reason = (f"chi1 has more than one populated state: {pops}. If plain MD stayed in one "
                        "state, it was trapped and its populations are wrong.")


def analyze(run_dir: Path, residues: list[ResidueKey], resnames: dict[ResidueKey, str],
            s: MetadSettings, temperature_k: float) -> list[EnhancedResult]:
    grid = np.linspace(-math.pi, math.pi, s.grid_points)
    replicas = sorted(int(p.stem.split("_")[1]) for p in (run_dir / "enhanced").glob("replica_*.json"))
    loaded = [(np.load(_paths(run_dir, i)["bias"]),
               np.load(_paths(run_dir, i)["bias_half"]) if _paths(run_dir, i)["bias_half"].exists() else None,
               np.load(_paths(run_dir, i)["chi"])) for i in replicas]
    results = []
    for j, key in enumerate(residues):
        pops, half, cross, fes = [], [], [], []
        for bias, bias_half, chi in loaded:
            f = free_energy(bias[j], s, temperature_k)
            fes.append(f.round(3).tolist())
            pops.append(populations(f, grid, temperature_k))
            if bias_half is not None:
                half.append(populations(free_energy(bias_half[j], s, temperature_k), grid, temperature_k))
            cross.append(count_transitions(assign_states(np.degrees(chi[:, j]))))
        r = EnhancedResult(key[0], key[1], resnames.get(key, "?"), pops, half, cross, fes)
        judge(r, s, temperature_k)
        results.append(r)
    return results


def format_results(results: list[EnhancedResult], ns: float, n_replicas: int) -> str:
    lines = [f"Enhanced sampling of chi1: {n_replicas} replicas x {ns:g} ns.", "",
             f"{'residue':<10}{'verdict':<18}{'g+':>6}{'t':>6}{'g-':>6}   crossings"]
    order = {"multiple_states": 0, "not_converged": 1, "single_state": 2}
    for r in sorted(results, key=lambda r: (order[r.verdict], r.resseq)):
        m = r.mean_populations
        lines.append(f"{r.label:<10}{r.verdict:<18}{m['g+']:>6.0%}{m['t']:>6.0%}{m['g-']:>6.0%}   "
                     + ",".join(map(str, r.crossings)))
    lines.append("")
    lines += [f"- {r.label}: {r.reason}" for r in sorted(results, key=lambda r: (order[r.verdict], r.resseq))]
    return "\n".join(lines)


def save_results(run_dir: Path, results: list[EnhancedResult], text: str) -> None:
    d = run_dir / "enhanced"
    (d / "results.txt").write_text(text + "\n")
    tmp = d / "results.tmp"
    tmp.write_text(json.dumps([asdict(r) for r in results], indent=1))
    tmp.replace(d / "results.json")
