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
    # Total equilibration before the first production part. It is split into
    # heating (1/6), restrained NPT (2/6), restraint release (2/6) and free NPT (1/6).
    equilibration_ps: float = 600.0
    restraint_kcal: float = 1.0   # Backbone (N, CA, C) restraint, kcal/mol/Å², during equilibration.
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


def make_simulation(files: SystemFiles, s: MDSettings, seed: int, extra_forces=()) -> app.Simulation:
    """NPT simulation with a Langevin thermostat and a Monte Carlo barostat."""
    system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(files.system))
    barostat = openmm.MonteCarloBarostat(s.pressure_bar * unit.bar, s.temperature_k * unit.kelvin)
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)
    for force in extra_forces:
        system.addForce(force)
    integrator = openmm.LangevinMiddleIntegrator(
        s.temperature_k * unit.kelvin, 1.0 / unit.picosecond, s.timestep_fs * unit.femtosecond)
    integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(s.platform) if s.platform else None
    return app.Simulation(files.topology, system, integrator, platform)


KCAL_A2_TO_KJ_NM2 = 418.4  # 1 kcal/mol/Å² in kJ/mol/nm²
HEAT_START_K = 5.0
HEAT_STEPS = 20
RELEASE_STEPS = 20


def backbone_atom_indices(topology: app.Topology) -> list[int]:
    keep = set(protein_atom_indices(topology))
    return [a.index for a in topology.atoms() if a.index in keep and a.name in ("N", "CA", "C")]


def equilibrate(files: SystemFiles, s: MDSettings, seed: int, log: Callable[[str], None], label: str):
    """Equilibrate from the built structure. Returns (positions, velocities, box vectors).

    1. Energy minimization with backbone restraints.
    2. NVT heating from 5 K to the target temperature, with backbone restraints.
    3. NPT with full backbone restraints.
    4. NPT while the restraint constant goes down to zero in steps.
    5. NPT without restraints.
    The restraint force exists only in this simulation; production starts from a
    new simulation without it, using the final positions, velocities and box.
    """
    t0 = time.time()
    steps_per_ps = round(1000.0 / s.timestep_fs)
    total = round(s.equilibration_ps * steps_per_ps)
    heat, restrained, release = total // 6, 2 * total // 6, 2 * total // 6
    free = total - heat - restrained - release

    restraint = openmm.CustomExternalForce("k_restraint*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k_restraint", s.restraint_kcal * KCAL_A2_TO_KJ_NM2)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    pos_nm = files.positions.value_in_unit(unit.nanometer)
    for i in backbone_atom_indices(files.topology):
        restraint.addParticle(i, list(pos_nm[i]))

    sim = make_simulation(files, s, seed, extra_forces=[restraint])
    barostat = next(f for f in sim.system.getForces() if isinstance(f, openmm.MonteCarloBarostat))
    frequency = barostat.getFrequency()
    barostat.setFrequency(0)  # NVT for minimization and heating.
    sim.context.reinitialize()
    box = files.topology.getPeriodicBoxVectors()
    if box is not None:
        sim.context.setPeriodicBoxVectors(*box)
    sim.context.setPositions(files.positions)
    sim.minimizeEnergy()

    integrator = sim.integrator
    integrator.setTemperature(HEAT_START_K * unit.kelvin)  # Heat from 5 K, not from the target.
    sim.context.setVelocitiesToTemperature(HEAT_START_K * unit.kelvin, seed)
    for i in range(1, HEAT_STEPS + 1):
        sim.step(max(1, heat // HEAT_STEPS))
        t = HEAT_START_K + (s.temperature_k - HEAT_START_K) * i / HEAT_STEPS
        integrator.setTemperature(t * unit.kelvin)

    barostat.setFrequency(frequency)  # Switch on pressure coupling.
    sim.context.reinitialize(preserveState=True)
    sim.step(restrained)
    k0 = s.restraint_kcal * KCAL_A2_TO_KJ_NM2
    for i in range(1, RELEASE_STEPS + 1):
        sim.context.setParameter("k_restraint", k0 * (1 - i / RELEASE_STEPS))
        sim.step(max(1, release // RELEASE_STEPS))
    sim.step(free)

    state = sim.context.getState(getPositions=True, getVelocities=True)
    log(f"{label}: minimized, heated to {s.temperature_k:g} K, equilibrated with backbone restraints "
        f"released step by step ({s.equilibration_ps:g} ps in total) in {time.time() - t0:.0f} s")
    return state.getPositions(), state.getVelocities(), state.getPeriodicBoxVectors()


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
    start_frame: tuple[Path, int] | None = None,
    full_interval_ps: float | None = None,
) -> None:
    """Run one part of one replica.

    By default, part k > 0 starts from the end state of part k - 1. With
    start_frame = (full-system trajectory, frame index), the part starts from
    that frame instead, with new random velocities (adaptive sampling).
    With full_interval_ps, the part also saves all atoms (with water) every
    full_interval_ps, so later parts can start from those frames.
    """
    s = settings
    resume = part > 0
    state_path = part_path(out_dir, replica, part, ".state.xml")
    steps_per_ps = round(1000.0 / s.timestep_fs)
    sim = make_simulation(files, s, seed)

    if start_frame is not None:
        import mdtraj as md

        dcd, frame = start_frame
        traj = md.load_frame(str(dcd), frame, top=str(out_dir / "solvated.pdb"))
        box = [openmm.Vec3(*v) for v in traj.unitcell_vectors[0]]
        sim.context.setPeriodicBoxVectors(*box)
        sim.context.setPositions(traj.xyz[0] * unit.nanometer)
        sim.context.setVelocitiesToTemperature(s.temperature_k * unit.kelvin, seed)
        log(f"replica {replica}: part {part} starts from frame {frame} of {dcd.name}")
    elif resume:
        sim.loadState(str(part_path(out_dir, replica, part - 1, ".state.xml")))
    else:
        positions, velocities, box = equilibrate(files, s, seed, log, f"replica {replica}")
        sim.context.setPeriodicBoxVectors(*box)
        sim.context.setPositions(positions)
        sim.context.setVelocities(velocities)

    interval = round(s.save_interval_ps * steps_per_ps)
    sim.reporters.append(app.DCDReporter(
        str(part_path(out_dir, replica, part, ".dcd")), interval,
        atomSubset=protein_atom_indices(files.topology)))
    if full_interval_ps:
        sim.reporters.append(app.DCDReporter(
            str(part_path(out_dir, replica, part, ".full.dcd")), round(full_interval_ps * steps_per_ps)))
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
