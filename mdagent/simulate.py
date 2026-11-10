"""Run independent MD replicas in segments that can be resumed.

Each replica is a chain of parts. Part 0 does energy minimization, a short
NPT equilibration, and then NPT production. Each later part starts from the
saved state (positions, velocities, box) at the end of the previous part.

Each part has its own trajectory, log and state file. If a part is stopped
halfway, running it again overwrites its files and starts from the same
state, so the chain stays consistent.

Replicas differ only in their random seed. Only protein atoms are saved to
the trajectory, to keep files small.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import openmm
from openmm import app, unit

from .build import BuiltSystem, protein_atom_indices


@dataclass
class MDSettings:
    equilibration_ps: float = 100.0
    timestep_fs: float = 2.0
    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    save_interval_ps: float = 5.0
    platform: str | None = None  # None lets OpenMM pick the fastest one.


@dataclass
class SystemFiles:
    """The system as stored on disk, so that a run can resume in a new process."""

    topology: app.Topology
    positions: unit.Quantity
    system: openmm.System


def write_inputs(built: BuiltSystem, out_dir: Path) -> None:
    """Save the solvated system and a protein-only topology for analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "solvated.pdb", "w") as f:
        app.PDBFile.writeFile(built.topology, built.positions, f, keepIds=True)
    with open(out_dir / "system.xml", "w") as f:
        f.write(openmm.XmlSerializer.serialize(built.system))

    keep = set(protein_atom_indices(built.topology))
    modeller = app.Modeller(built.topology, built.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.index not in keep])
    with open(out_dir / "protein.pdb", "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)


def load_inputs(out_dir: Path) -> SystemFiles:
    pdb = app.PDBFile(str(out_dir / "solvated.pdb"))
    system = openmm.XmlSerializer.deserialize((out_dir / "system.xml").read_text())
    return SystemFiles(pdb.topology, pdb.positions, system)


def part_path(out_dir: Path, replica: int, part: int, suffix: str) -> Path:
    return out_dir / f"replica_{replica}_part{part:03d}{suffix}"


def run_segment(
    files: SystemFiles,
    out_dir: Path,
    replica: int,
    part: int,
    seed: int,
    production_ns: float,
    settings: MDSettings,
    log: Callable[[str], None] = print,
) -> None:
    """Run one part of one replica. Part k > 0 starts from the end state of part k - 1."""
    s = settings
    resume = part > 0
    state_path = part_path(out_dir, replica, part, ".state.xml")
    steps_per_ps = round(1000.0 / s.timestep_fs)

    system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(files.system))
    barostat = openmm.MonteCarloBarostat(s.pressure_bar * unit.bar, s.temperature_k * unit.kelvin)
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)
    integrator = openmm.LangevinMiddleIntegrator(
        s.temperature_k * unit.kelvin, 1.0 / unit.picosecond, s.timestep_fs * unit.femtosecond)
    integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(s.platform) if s.platform else None
    sim = app.Simulation(files.topology, system, integrator, platform)

    if resume:
        sim.loadState(str(part_path(out_dir, replica, part - 1, ".state.xml")))
    else:
        t0 = time.time()
        sim.context.setPositions(files.positions)
        sim.minimizeEnergy()
        sim.context.setVelocitiesToTemperature(s.temperature_k * unit.kelvin, seed)
        sim.step(round(s.equilibration_ps * steps_per_ps))
        log(f"replica {replica}: minimized and equilibrated ({s.equilibration_ps:g} ps) "
            f"in {time.time() - t0:.0f} s")

    interval = round(s.save_interval_ps * steps_per_ps)
    sim.reporters.append(app.DCDReporter(
        str(part_path(out_dir, replica, part, ".dcd")), interval,
        atomSubset=protein_atom_indices(files.topology)))
    sim.reporters.append(app.StateDataReporter(
        str(part_path(out_dir, replica, part, ".csv")), interval,
        time=True, potentialEnergy=True, temperature=True, density=True, speed=True))

    n_steps = round(production_ns * 1000 * steps_per_ps)
    n_steps -= n_steps % interval  # Keep whole frames so part lengths add up exactly.
    chunk = max(interval, (n_steps // 5) // interval * interval)
    t1 = time.time()
    done = 0
    while done < n_steps:
        step = min(chunk, n_steps - done)
        sim.step(step)
        done += step
        ns_per_day = (done / steps_per_ps / 1000) / ((time.time() - t1) / 86400)
        log(f"replica {replica}: {done / n_steps:4.0%} of part {part} ({production_ns:g} ns) "
            f"({ns_per_day:.0f} ns/day)")

    tmp = state_path.with_suffix(".tmp")
    sim.saveState(str(tmp))
    os.replace(tmp, state_path)  # The state file appears only when the part is complete.
