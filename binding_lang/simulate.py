"""Run several independent MD replicas of a built system.

Each replica: energy minimization, short NPT equilibration, NPT production.
Replicas differ only in their random seed (initial velocities and thermostat noise).
Only protein atoms are saved to the trajectory, to keep files small.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import openmm
from openmm import app, unit

from .build import BuiltSystem


@dataclass
class MDSettings:
    production_ns: float = 1.0
    equilibration_ps: float = 100.0
    timestep_fs: float = 2.0
    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    save_interval_ps: float = 2.0
    platform: str | None = None  # None lets OpenMM pick the fastest one.


def write_inputs(built: BuiltSystem, out_dir: Path) -> None:
    """Save the solvated system and a protein-only topology for analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "solvated.pdb", "w") as f:
        app.PDBFile.writeFile(built.topology, built.positions, f, keepIds=True)
    with open(out_dir / "system.xml", "w") as f:
        f.write(openmm.XmlSerializer.serialize(built.system))

    protein = built.protein_atom_indices()
    modeller = app.Modeller(built.topology, built.positions)
    keep = set(protein)
    modeller.delete([a for a in modeller.topology.atoms() if a.index not in keep])
    with open(out_dir / "protein.pdb", "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)


def run_replica(
    built: BuiltSystem,
    out_dir: Path,
    replica: int,
    seed: int,
    settings: MDSettings,
    log: Callable[[str], None] = print,
) -> Path:
    s = settings
    dt = s.timestep_fs * unit.femtosecond
    steps_per_ps = round(1000.0 / s.timestep_fs)

    system = copy.deepcopy(built.system)
    barostat = openmm.MonteCarloBarostat(s.pressure_bar * unit.bar, s.temperature_k * unit.kelvin)
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)

    integrator = openmm.LangevinMiddleIntegrator(s.temperature_k * unit.kelvin, 1.0 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(s.platform) if s.platform else None
    sim = app.Simulation(built.topology, system, integrator, platform)
    sim.context.setPositions(built.positions)

    t0 = time.time()
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(s.temperature_k * unit.kelvin, seed)
    sim.step(round(s.equilibration_ps * steps_per_ps))
    log(f"replica {replica}: minimized and equilibrated ({s.equilibration_ps:g} ps) "
        f"in {time.time() - t0:.0f} s")

    dcd = out_dir / f"replica_{replica}.dcd"
    interval = round(s.save_interval_ps * steps_per_ps)
    sim.reporters.append(app.DCDReporter(str(dcd), interval, atomSubset=built.protein_atom_indices()))
    sim.reporters.append(app.StateDataReporter(
        str(out_dir / f"replica_{replica}.csv"), interval,
        time=True, potentialEnergy=True, temperature=True, density=True, speed=True,
    ))

    n_steps = round(s.production_ns * 1000 * steps_per_ps)
    chunk = max(1, n_steps // 10)
    t1 = time.time()
    done = 0
    while done < n_steps:
        step = min(chunk, n_steps - done)
        sim.step(step)
        done += step
        elapsed = time.time() - t1
        ns_per_day = (done / steps_per_ps / 1000) / (elapsed / 86400)
        log(f"replica {replica}: {done / n_steps:4.0%} of {s.production_ns:g} ns "
            f"({ns_per_day:.1f} ns/day)")
    return dcd


def run_all(
    built: BuiltSystem,
    out_dir: str | Path,
    n_replicas: int,
    settings: MDSettings,
    base_seed: int = 1,
    log: Callable[[str], None] = print,
) -> list[Path]:
    out_dir = Path(out_dir)
    write_inputs(built, out_dir)
    paths = []
    for i in range(n_replicas):
        paths.append(run_replica(built, out_dir, i, base_seed + i, settings, log))
    meta = {
        "n_replicas": n_replicas,
        "base_seed": base_seed,
        "settings": settings.__dict__,
        "build_notes": built.notes,
        "trajectories": [p.name for p in paths],
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))
    return paths
