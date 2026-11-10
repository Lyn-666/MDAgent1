"""Build a simulation-ready system from a protein PDB file.

Steps: remove ligands and crystal waters, add missing atoms and internal
missing residues, add hydrogens at a given pH, add a water box and ions.

Limitations of this first version:
- Ligands are removed. The simulation is of the protein only (apo).
- Protonation uses PDBFixer's default rules at the given pH. It does not
  predict pKa shifts inside the protein.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import openmm
from openmm import app, unit
from pdbfixer import PDBFixer
from pdbfixer.pdbfixer import proteinResidues

FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3p.xml")


@dataclass
class BuiltSystem:
    topology: app.Topology
    positions: unit.Quantity
    system: openmm.System
    notes: list[str] = field(default_factory=list)


_PROTEIN = set(proteinResidues)


def protein_atom_indices(topology: app.Topology) -> list[int]:
    return [a.index for a in topology.atoms() if a.residue.name in _PROTEIN]


def build_system(
    pdb_path: str | Path,
    ph: float = 7.0,
    padding_nm: float = 1.0,
    ionic_strength_molar: float = 0.15,
    hydrogen_mass_amu: float | None = None,
) -> BuiltSystem:
    notes: list[str] = []
    fixer = PDBFixer(filename=str(pdb_path))

    removed = sorted({r.name for r in fixer.topology.residues()
                      if r.name not in _PROTEIN and r.name != "HOH"})
    fixer.removeHeterogens(keepWater=False)
    if removed:
        notes.append(f"Removed non-protein molecules: {', '.join(removed)}.")

    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    for key in list(fixer.missingResidues):
        chain = chains[key[0]]
        # Do not model missing residues at chain ends. They are usually disordered.
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]
    n_missing_res = sum(len(v) for v in fixer.missingResidues.values())
    if n_missing_res:
        notes.append(f"Added {n_missing_res} missing internal residues.")

    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        names = ", ".join(sorted({r.name for r, _ in fixer.nonstandardResidues}))
        notes.append(f"Replaced nonstandard residues: {names}.")
    fixer.replaceNonstandardResidues()

    fixer.findMissingAtoms()
    n_missing_atoms = sum(len(v) for v in fixer.missingAtoms.values())
    if n_missing_atoms:
        notes.append(f"Added {n_missing_atoms} missing heavy atoms.")
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    notes.append(f"Added hydrogens at pH {ph}.")

    forcefield = app.ForceField(*FORCEFIELD_FILES)
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        padding=padding_nm * unit.nanometer,
        ionicStrength=ionic_strength_molar * unit.molar,
        positiveIon="Na+",
        negativeIon="Cl-",
    )
    n_water = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
    notes.append(
        f"Added a water box ({padding_nm} nm padding, {n_water} waters) "
        f"and ions to {ionic_strength_molar} M NaCl."
    )

    kwargs = {}
    if hydrogen_mass_amu:
        kwargs["hydrogenMass"] = hydrogen_mass_amu * unit.amu
        notes.append(f"Hydrogen mass set to {hydrogen_mass_amu} amu.")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        **kwargs,
    )
    notes.append(
        f"Force field: Amber14 + TIP3P. System has {system.getNumParticles()} atoms."
    )
    return BuiltSystem(modeller.topology, modeller.positions, system, notes)
