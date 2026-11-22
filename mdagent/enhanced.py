"""Enhanced sampling of dihedral angles with well-tempered metadynamics.

Plain MD cannot push a side chain or a backbone over a high barrier. Metadynamics adds a
bias potential along a dihedral that grows wherever the system has been, so it is pushed
over the barriers. In the well-tempered form, the bias grows more slowly where it is
already large.

What is biased (`kind`):
- "chi1": the side-chain chi1 of each residue (one dihedral per residue);
- "backbone": the two dihedrals of each backbone item (rama:A:110: phi and psi of residue 110;
  turn:A:52: psi of 52 and phi of 53), each with its own one-dimensional bias, in the order of
  the items. A sum of one-dimensional
  biases is a separable bias; reweighting stays exact (the same replay as for several
  coupled side chains, tested in tests/test_reweight.py). Biased sampling changes the
  kinetics, so transition rates and dwell times must still come from plain MD.

All dihedrals are biased in one simulation, each with its own one-dimensional bias. All
biases live in one OpenMM CustomCVForce, so the cost is small. The saved samples
(replica_<r>_chi.npy; the name is kept for older runs) have one column per biased dihedral.

Each replica starts from the final state of a plain MD replica. It runs in
chunks. After each chunk it saves the protein trajectory of that chunk, the
state, the bias and the chi1 samples, so a stopped run can resume.

This module runs the simulation and saves the data. reweight.py analyzes it.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mdtraj as md
import numpy as np
import openmm
from openmm import app

from .build import protein_atom_indices
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
    pace_ps: float = 1.0          # Time between Gaussians (and between saved chi1 samples).
    grid_points: int = 361        # Grid on chi1 from -180° to 180° (both ends included).
    chunk_ns: float = 1.0         # Save progress after each chunk.
    save_interval_ps: float = 5.0  # Time between saved trajectory frames.
    # At most this many residues are biased at once (the slowest ones). Biasing many side
    # chains at the same time pushes hard on the protein and can unfold a small one.
    max_residues: int = 10

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


def dihedral_quads(topology: openmm.app.Topology, residues: list[ResidueKey], kind: str = "chi1") -> list[tuple]:
    """Atom quadruples of the biased dihedrals, in the order of the saved columns (see the module notes)."""
    if kind == "chi1":
        quads = chi1_quads(topology, residues)
        return [quads[k] for k in residues]
    if kind == "backbone":
        # `residues` are backbone item specs here (rama:A:110, turn:A:52): two dihedrals each, in their order.
        from .basins import BasinSet, dihedral_atoms

        top = md.Topology.from_openmm(topology)
        return [tuple(q) for spec in residues for q in dihedral_atoms(top, BasinSet.parse(spec))]
    raise ValueError(f"unknown kind of enhanced sampling {kind!r}")


def deposit(bias: np.ndarray, grid: np.ndarray, x: np.ndarray, s: MetadSettings, kt: float) -> None:
    """Add one well-tempered Gaussian per residue at the chi1 values x (radians), in place.

    The height shrinks where the bias is already large: h * exp(-V(x) / (kT (gamma - 1))).
    The analysis (reweight.py) replays exactly these steps to rebuild the bias at any time.
    """
    gamma = s.bias_factor
    sigma = math.radians(s.sigma_deg)
    for i, xi in enumerate(x):
        height = s.height_kj * math.exp(-float(np.interp(xi, grid, bias[i])) / (kt * (gamma - 1)))
        d = (grid - xi + math.pi) % (2 * math.pi) - math.pi
        bias[i] += height * np.exp(-d * d / (2 * sigma ** 2))
    bias[:, -1] = bias[:, 0]  # Both ends are the same angle.


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
        deposit(self.bias, self.grid, x, self.s, self.kt)

    def push(self, context) -> None:
        for i, table in enumerate(self.tables):
            table.setFunctionParameters(self.bias[i].tolist(), -math.pi, math.pi)
        self.force.updateParametersInContext(context)


def ns_done(run_dir: Path, replica: int) -> float:
    """Enhanced sampling already done by a replica (ns)."""
    path = _paths(run_dir, replica)["progress"]
    return json.loads(path.read_text())["ns_done"] if path.exists() else 0.0


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
    until_ns: float | None = None,
    kind: str = "chi1",
) -> None:
    """Run one replica up to `until_ns` (default: its full length), one chunk at a time.

    kind: "chi1" (side chains) or "backbone" (phi and psi of each residue)."""
    p = _paths(run_dir, replica)
    p["dir"].mkdir(exist_ok=True)
    quads = dihedral_quads(files.topology, residues, kind)
    progress = json.loads(p["progress"].read_text()) if p["progress"].exists() else {"ns_done": 0.0}
    end = min(s.length_ns, until_ns if until_ns is not None else s.length_ns)
    if progress["ns_done"] >= end - 1e-9:
        if progress["ns_done"] >= s.length_ns - 1e-9:
            log(f"enhanced replica {replica}: already complete ({progress['ns_done']:g} ns).")
        return
    # Each start gets its own random seed, so the thermostat noise does not repeat.
    seed += 7919 * round(progress["ns_done"] * 1000)

    resuming = progress["ns_done"] > 0
    bias = np.load(p["bias"]) if resuming else None
    chi = list(np.load(p["chi"])) if resuming else []
    biases = ChiBiases(quads, s, md_settings.temperature_k, bias)
    sim = make_simulation(files, md_settings, seed, extra_forces=[biases.force])
    sim.loadState(str(p["state"] if resuming else start_state))
    biases.push(sim.context)

    steps_per_ps = round(1000.0 / md_settings.timestep_fs)
    pace = round(s.pace_ps * steps_per_ps)
    t0 = time.time()
    ns_start = progress["ns_done"]
    frame_interval = round(s.save_interval_ps * steps_per_ps)
    while progress["ns_done"] < end - 1e-9:
        chunk_ns = min(s.chunk_ns, end - progress["ns_done"])
        n_pace = round(chunk_ns * 1000 / s.pace_ps)
        # One trajectory file per chunk. If a chunk is stopped halfway, running it
        # again overwrites its file.
        chunk_index = round(progress["ns_done"] / s.chunk_ns)
        sim.reporters = [app.DCDReporter(
            str(p["dir"] / f"replica_{replica}_chunk{chunk_index:03d}.dcd"), frame_interval,
            atomSubset=protein_atom_indices(files.topology))]
        report_every = max(1, round(0.1 * 1000 / s.pace_ps))  # Report progress every 0.1 ns.
        for k in range(1, n_pace + 1):
            sim.step(pace)
            x = biases.values(sim.context)
            chi.append(x)
            biases.deposit(x)
            biases.push(sim.context)
            if k % report_every == 0 and k < n_pace:
                now = progress["ns_done"] + k * s.pace_ps / 1000
                speed = (now - ns_start) / ((time.time() - t0) / 86400)
                log(f"enhanced replica {replica}: {now:.1f} of {s.length_ns:g} ns ({speed:.0f} ns/day, "
                    f"{time.strftime('%m-%d %H:%M')})")
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
            f"({speed:.0f} ns/day, {time.strftime('%m-%d %H:%M')})")
