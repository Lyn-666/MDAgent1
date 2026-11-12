"""Enhanced sampling of side-chain chi1 angles with well-tempered metadynamics.

Plain MD cannot push a side chain over a high barrier. Metadynamics adds a
bias potential along chi1 that grows wherever the side chain has been, so the
side chain is pushed over the barriers. In the well-tempered form, the bias
grows more slowly where it is already large.

All residues are biased in one simulation. Each chi1 has its own
one-dimensional bias. All biases live in one OpenMM CustomCVForce, so the
cost is small.

Each replica starts from the final state of a plain MD replica. It runs in
chunks. After each chunk it saves the protein trajectory of that chunk, the
state, the bias and the chi1 samples, so a stopped run can resume.

This module only runs the simulation and saves the data. It does not analyze it.
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
    frame_interval = round(s.save_interval_ps * steps_per_ps)
    while progress["ns_done"] < s.length_ns - 1e-9:
        chunk_ns = min(s.chunk_ns, s.length_ns - progress["ns_done"])
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
                log(f"enhanced replica {replica}: {now:.1f} of {s.length_ns:g} ns ({speed:.0f} ns/day)")
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
