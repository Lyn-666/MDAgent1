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

Populations are computed by reweighting with each residue's own bias and the
biases of the side chains coupled to it (see the analysis section below). A result counts as converged only when both
replicas and both halves of each replica agree. If not, the agent extends
the run, up to max_length_ns.
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
    length_ns: float = 20.0       # Per replica, for the first pass.
    max_length_ns: float = 60.0   # The agent extends unconverged runs up to this length.
    extend_ns: float = 10.0       # Each extension adds this much per replica.
    n_replicas: int = 2
    bias_factor: float = 10.0     # gamma. Larger pushes harder over barriers.
    height_kj: float = 1.2        # Initial height of each Gaussian.
    sigma_deg: float = 12.0       # Width of each Gaussian.
    pace_ps: float = 1.0          # Time between Gaussians (and between chi1 samples).
    grid_points: int = 361        # Grid on chi1 from -180° to 180° (both ends included).
    chunk_ns: float = 1.0         # Save progress after each chunk.
    transient_fraction: float = 0.2     # Skip this first part of each replica in the analysis.
    single_state_cutoff: float = 0.95   # Main state above this population: one stable state.
    agreement_kj: float = 2.5           # Max free energy difference between estimates (about 1 kT).
    population_floor: float = 0.05      # States below this population count as negligible.
    min_crossings: int = 4               # Min chi1 state changes per replica under bias.
    coupling_threshold: float = 0.2     # Cramer's V above this: two side chains affect each other.
    min_effective_samples: float = 200.0  # Fewer effective samples than this: estimate not trusted.

    @classmethod
    def from_dict(cls, d: dict) -> "MetadSettings":
        """Ignore keys from older versions."""
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


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
#
# Populations come from reweighting. Every pace_ps we saved the chi1 angle of
# every biased residue. For residue j, each saved moment gets the weight
# exp(V / kT), where V is the sum of the final biases of residue j and of the
# residues coupled to it. Coupled means that their chi1 states are
# statistically dependent (Cramer's V above coupling_threshold). We do not use
# all biases: with eight large biases summed, the weights differ by many orders
# of magnitude, and only a handful of samples would count. This happened in
# the first 181L test. The first transient_fraction of each replica is skipped,
# because the bias was still far from its final shape.
#
# The free energy curve along chi1 still comes from each residue's own bias:
# F(chi) = -(gamma / (gamma - 1)) * V(chi). It is saved for plotting.


def free_energy(bias: np.ndarray, s: MetadSettings, temperature_k: float) -> np.ndarray:
    """F(chi) in kJ/mol, with minimum 0, from a well-tempered bias."""
    gamma = s.bias_factor
    f = -gamma / (gamma - 1) * bias
    return f - f.min()


def nearest_state(chi_rad: np.ndarray) -> np.ndarray:
    deg = np.degrees(chi_rad)
    dist = np.stack([angular_distance_deg(deg, c) for c in STATE_CENTERS_DEG.values()])
    return dist.argmin(axis=0)


def reweighted_populations(chi: np.ndarray, bias: np.ndarray, j: int, grid: np.ndarray,
                           temperature_k: float, use: list[int] | None = None
                           ) -> tuple[dict[str, float], float]:
    """Populations of residue j from samples chi (n_samples x n_residues) and final biases.

    The weight of each sample is exp(V / kT), where V is the sum of the biases of the
    residues in `use` (default: residue j only). Returns the populations and the
    effective number of samples, (sum w)^2 / sum w^2.
    """
    kt = KB_KJ * temperature_k
    use = [j] if use is None else use
    v = sum(np.interp(chi[:, i], grid, bias[i]) for i in use)
    w = np.exp((v - v.max()) / kt)
    states = nearest_state(chi[:, j])
    total = w.sum()
    pops = {name: float(w[states == k].sum() / total) for k, name in enumerate(STATE_NAMES)}
    ess = float(total ** 2 / (w ** 2).sum())
    return pops, ess


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Strength of the statistical dependence between two state sequences (0: none, 1: full)."""
    table = np.zeros((3, 3))
    np.add.at(table, (a, b), 1)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if min(table.shape) < 2:
        return 0.0
    n = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(math.sqrt(chi2 / (n * (min(table.shape) - 1))))


def coupled_partners(chi: np.ndarray, j: int, threshold: float) -> list[int]:
    """Other residues whose chi1 state depends on the chi1 state of residue j.

    Reweighting residue j must also undo the biases of these residues, because
    their bias changes the environment of residue j. Including every residue
    makes the weights so uneven that only a few samples count.
    """
    states = [nearest_state(chi[:, i]) for i in range(chi.shape[1])]
    return [k for k in range(chi.shape[1])
            if k != j and cramers_v(states[j], states[k]) > threshold]


def state_free_energies(p: dict[str, float], floor: float, temperature_k: float) -> dict[str, float]:
    """-kT ln(p), relative to the main state. Populations below floor are raised to floor."""
    kt = KB_KJ * temperature_k
    top = max(p.values())
    return {n: -kt * math.log(max(p[n], floor) / top) for n in STATE_NAMES}


@dataclass
class EnhancedResult:
    chain: str
    resseq: int
    resname: str
    populations: list[dict[str, float]]          # Per replica, after the transient.
    half_populations: list[list[dict[str, float]]]  # Per replica: [first half, second half].
    crossings: list[int]
    effective_samples: list[float] = field(default_factory=list)
    coupled_with: list[str] = field(default_factory=list)
    free_energy: list[list[float]] = field(repr=False, default_factory=list)  # Per replica, on the grid.
    spread_kj: float = 0.0
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
    estimates = list(r.populations) + [h for halves in r.half_populations for h in halves]
    fes = [state_free_energies(p, s.population_floor, temperature_k) for p in estimates]
    r.spread_kj = max(max(f[n] for f in fes) - min(f[n] for f in fes) for n in STATE_NAMES)
    mean = r.mean_populations
    pops = ", ".join(f"{n} {mean[n]:.0%}" for n in STATE_NAMES)

    if r.effective_samples and min(r.effective_samples) < s.min_effective_samples:
        r.verdict = "not_converged"
        r.reason = (f"Too few effective samples after reweighting ({min(r.effective_samples):.0f}; "
                    f"at least {s.min_effective_samples:.0f} are needed). Current estimate: {pops}.")
    elif min(r.crossings) < s.min_crossings:
        r.verdict = "not_converged"
        r.reason = (f"Even with the bias, chi1 changed state only {r.crossings} times per replica. "
                    "The enhanced sampling needs to run longer.")
    elif r.spread_kj > s.agreement_kj:
        r.verdict = "not_converged"
        r.reason = (f"The replicas, or the two halves of a replica, disagree by up to {r.spread_kj:.1f} kJ/mol "
                    f"(the limit is {s.agreement_kj:g}). Current estimate: {pops}.")
    else:
        main = max(mean, key=mean.get)
        others = [n for n in STATE_NAMES if n != main]
        gaps = {n: -kt * math.log(max(mean[n], 1e-6) / mean[main]) for n in others}
        agree = f"Replicas and halves agree within {r.spread_kj:.1f} kJ/mol."
        if mean[main] >= s.single_state_cutoff:
            r.verdict = "single_state"
            r.reason = (f"chi1 has one stable state: {main} ({mean[main]:.0%}). The other states are "
                        + " and ".join(f"{n} {gaps[n]:.1f} kJ/mol" for n in others)
                        + f" higher. Plain MD was not trapped; this side chain is sampled. {agree}")
        else:
            r.verdict = "multiple_states"
            minor = " and ".join(f"{n} {gaps[n]:.1f} kJ/mol" for n in others if mean[n] >= 0.01)
            r.reason = (f"chi1 has more than one populated state: {pops}. The minor states are {minor} "
                        f"above {main}. If plain MD stayed in {main}, it missed {1 - mean[main]:.0%} of the "
                        f"population. {agree}")


def analyze(run_dir: Path, residues: list[ResidueKey], resnames: dict[ResidueKey, str],
            s: MetadSettings, temperature_k: float) -> list[EnhancedResult]:
    grid = np.linspace(-math.pi, math.pi, s.grid_points)
    replicas = sorted(int(p.stem.split("_")[1]) for p in (run_dir / "enhanced").glob("replica_*.json"))
    loaded = [(np.load(_paths(run_dir, i)["bias"]), np.load(_paths(run_dir, i)["chi"])) for i in replicas]
    posts = [chi[int(len(chi) * s.transient_fraction):] for _, chi in loaded]
    all_post = np.concatenate(posts)
    labels = [f"{resnames.get(k, '?')}{k[1]}" for k in residues]
    results = []
    for j, key in enumerate(residues):
        partners = coupled_partners(all_post, j, s.coupling_threshold)
        use = [j] + partners
        pops, halves, cross, ess, fes = [], [], [], [], []
        for (bias, chi), post in zip(loaded, posts):
            mid = len(post) // 2
            p, e = reweighted_populations(post, bias, j, grid, temperature_k, use)
            pops.append(p)
            ess.append(round(e, 1))
            halves.append([reweighted_populations(post[:mid], bias, j, grid, temperature_k, use)[0],
                           reweighted_populations(post[mid:], bias, j, grid, temperature_k, use)[0]])
            cross.append(count_transitions(assign_states(np.degrees(chi[:, j]))))
            fes.append(free_energy(bias[j], s, temperature_k).round(3).tolist())
        r = EnhancedResult(key[0], key[1], resnames.get(key, "?"), pops, halves, cross, ess,
                           [labels[k] for k in partners], fes)
        judge(r, s, temperature_k)
        results.append(r)
    return results


def format_results(results: list[EnhancedResult], ns: float, n_replicas: int) -> str:
    lines = [f"Enhanced sampling of chi1: {n_replicas} replicas x {ns:g} ns. Each residue is reweighted "
             "with its own bias and the biases of the side chains coupled to it.", "",
             f"{'residue':<10}{'verdict':<18}{'g+':>6}{'t':>6}{'g-':>6}{'spread':>10}{'eff. samples':>15}"
             "   coupled with"]
    order = {"multiple_states": 0, "not_converged": 1, "single_state": 2}
    ranked = sorted(results, key=lambda r: (order[r.verdict], r.resseq))
    for r in ranked:
        m = r.mean_populations
        ess = f"{min(r.effective_samples):.0f}" if r.effective_samples else "-"
        lines.append(f"{r.label:<10}{r.verdict:<18}{m['g+']:>6.0%}{m['t']:>6.0%}{m['g-']:>6.0%}"
                     f"{r.spread_kj:>7.1f} kJ{ess:>15}   {', '.join(r.coupled_with) or '-'}")
    lines.append("")
    lines.append("spread: the largest free energy difference between the replicas and the halves of each "
                 "replica. Below 2.5 kJ/mol counts as converged.")
    lines.append("eff. samples: the smallest effective number of samples after reweighting (replica minimum). "
                 "Below 200, the estimate is not trusted.")
    lines.append("coupled with: biased side chains whose chi1 state depends on this one.")
    lines.append("")
    lines += [f"- {r.label}: {r.reason}" for r in ranked]
    return "\n".join(lines)


def save_results(run_dir: Path, results: list[EnhancedResult], text: str) -> None:
    d = run_dir / "enhanced"
    (d / "results.txt").write_text(text + "\n")
    tmp = d / "results.tmp"
    tmp.write_text(json.dumps([asdict(r) for r in results], indent=1))
    tmp.replace(d / "results.json")
