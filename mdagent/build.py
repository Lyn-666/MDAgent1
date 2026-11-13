"""Build a simulation-ready system from a protein PDB file.

All choices are in one SetupOptions object. Each choice has a default.
The command line fills SetupOptions now; a chat interface can fill the same
object later.

Steps:
1. Remove ligands, ions and crystal waters.
2. Report missing residues. Model missing residues inside a chain. Do not
   model missing residues at chain ends (they are usually disordered).
3. Add missing heavy atoms. Remove any hydrogens in the input file.
4. Choose protonation states: defaults at the given pH, then PROPKA
   predictions (only if requested), then the user's manual choices.
5. Add hydrogens, a periodic water box, and ions.
6. Create the system with the nonbonded settings that belong to the force field.

Limitation: ligands are removed. The simulation is of the protein only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import openmm
from openmm import app, unit
from pdbfixer import PDBFixer
from pdbfixer.pdbfixer import proteinResidues

from . import protonation

_PROTEIN = set(proteinResidues)


@dataclass(frozen=True)
class ForceFieldPreset:
    description: str
    files: tuple[str, ...]
    solvent_geometry: str        # Water model name that Modeller.addSolvent knows.
    cutoff_nm: float
    switch_nm: float | None      # Start of the Lennard-Jones switching function.
    dispersion_correction: bool  # Long-range correction for Lennard-Jones.


FORCEFIELDS = {
    "charmm36m": ForceFieldPreset(
        "CHARMM36m protein + CHARMM TIP3P water (July 2024 CHARMM release)",
        ("charmm36_2024.xml", "charmm36_2024/water.xml"),
        "tip3p", cutoff_nm=1.2, switch_nm=1.0, dispersion_correction=False),
    "amber14": ForceFieldPreset(
        "Amber ff14SB protein + TIP3P water",
        ("amber14-all.xml", "amber14/tip3p.xml"),
        "tip3p", cutoff_nm=1.0, switch_nm=None, dispersion_correction=True),
    "amber19": ForceFieldPreset(
        "Amber ff19SB protein + OPC water",
        ("amber19-all.xml", "amber19/opc.xml"),
        # OPC is a 4-site water. Modeller places 4-site TIP4P-Ew waters; the
        # OPC force field then assigns its own parameters.
        "tip4pew", cutoff_nm=1.0, switch_nm=None, dispersion_correction=True),
}
CATIONS = ("Na+", "K+")
BOX_SHAPES = ("cube", "dodecahedron", "octahedron")
LONG_GAP = 5          # A modeled gap longer than this gets a warning.
NEAR_TARGET = 4       # A modeled gap within this many residues of the target gets a warning.
MIN_BOX_FACTOR = 1.1  # The box must be at least 1.1 x (2 x cutoff) wide.


@dataclass
class SetupOptions:
    forcefield: str = "charmm36m"
    ph: float = 7.0
    ion_conc_molar: float = 0.15
    cation: str = "Na+"
    padding_nm: float = 1.0
    box_shape: str = "cube"
    check_protonation: bool = False
    protonation: dict[str, str] = field(default_factory=dict)  # "A:31" -> "HIP"
    hydrogen_mass_amu: float | None = None
    # Structure check (check.py) choices.
    chains: list[str] | None = None      # Keep only these protein chains. None: keep all.
    keep_tags: bool = False              # Keep tag residues at the chain ends.
    uniprot: str | None = None           # Reference sequence when the file has none.
    accept_problems: bool = False        # Build even if the structure check found problems.

    def validate(self) -> None:
        if self.forcefield not in FORCEFIELDS:
            raise ValueError(f"Unknown force field {self.forcefield!r}. Choose from {list(FORCEFIELDS)}.")
        if self.cation not in CATIONS:
            raise ValueError(f"Unknown cation {self.cation!r}. Choose from {list(CATIONS)}.")
        if self.box_shape not in BOX_SHAPES:
            raise ValueError(f"Unknown box shape {self.box_shape!r}. Choose from {list(BOX_SHAPES)}.")
        if not 0 <= self.ion_conc_molar <= 2:
            raise ValueError("Ion concentration must be between 0 and 2 M.")
        if not 0 < self.ph < 14:
            raise ValueError("pH must be between 0 and 14.")
        if self.padding_nm < 0.8:
            raise ValueError("Padding below 0.8 nm lets the protein interact with its periodic image.")


@dataclass
class BuiltSystem:
    topology: app.Topology
    positions: unit.Quantity
    system: openmm.System
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = list(self.notes)
        if self.warnings:
            lines.append("")
            lines += ["WARNING: " + w for w in self.warnings]
        return "\n".join(lines)


def _min_box_width_nm(vectors) -> float:
    """Smallest distance between opposite faces of a (possibly triclinic) box."""
    import numpy as np

    a, b, c = (np.array(v.value_in_unit(unit.nanometer)) for v in vectors)
    volume = abs(np.dot(a, np.cross(b, c)))
    return min(volume / np.linalg.norm(np.cross(b, c)),
               volume / np.linalg.norm(np.cross(c, a)),
               volume / np.linalg.norm(np.cross(a, b)))


def _amber14() -> app.ForceField:
    return app.ForceField(*FORCEFIELDS["amber14"].files)


def protein_atom_indices(topology: app.Topology) -> list[int]:
    """Atoms of amino acid residues. A residue counts as an amino acid when it has
    backbone atoms N, CA and C. This also works for Amber names such as HIE, CYX or ASH."""
    out = []
    for res in topology.residues():
        names = {a.name for a in res.atoms()}
        if res.name in _PROTEIN or {"N", "CA", "C"} <= names:
            out.extend(a.index for a in res.atoms())
    return sorted(out)


def _gap_report(fixer: PDBFixer, target: set[tuple[str, int]], notes, warnings) -> None:
    """Describe missing residues, and drop the ones at chain ends."""
    chains = list(fixer.topology.chains())
    for key in sorted(fixer.missingResidues):
        names = fixer.missingResidues[key]
        chain = chains[key[0]]
        residues = list(chain.residues())
        n = len(names)
        if key[1] == 0:
            first = int(residues[0].id) - n
            where, modeled = "N-terminal end", False
        elif key[1] == len(residues):
            first = int(residues[-1].id) + 1
            where, modeled = "C-terminal end", False
        else:
            first = int(residues[key[1] - 1].id) + 1
            where, modeled = "inside the chain", True
        last = first + n - 1
        span = f"chain {chain.id} residues {first}-{last} ({n} residues, {where})"
        if not modeled:
            notes.append(f"Missing {span}: not modeled, because chain ends are usually disordered.")
            del fixer.missingResidues[key]
            continue
        notes.append(f"Missing {span}: modeled by PDBFixer.")
        near = any(c == chain.id and first - NEAR_TARGET <= r <= last + NEAR_TARGET for c, r in target)
        if n > LONG_GAP or near:
            reason = "it is long" if n > LONG_GAP else "it is near the target region"
            warnings.append(
                f"The modeled gap in {span} may be unreliable because {reason}. "
                "Consider a structure without this gap, for example an AlphaFold model.")


def _delete_residues(fixer: PDBFixer, keys: list[tuple[str, int, str]]) -> None:
    """Delete residues (chain, number, insertion code), for example tags found by the structure check."""
    wanted = {(c, n, i) for c, n, i in keys}
    modeller = app.Modeller(fixer.topology, fixer.positions)
    gone = [r for r in modeller.topology.residues()
            if (r.chain.id, int(r.id), (r.insertionCode or "").strip()) in wanted]
    modeller.delete(gone)
    fixer.topology, fixer.positions = modeller.topology, modeller.positions


def _chain_break_check(fixer: PDBFixer, warnings) -> None:
    """Find chain breaks that PDBFixer did not report.

    PDBFixer finds missing residues only from the SEQRES records. Without
    them, a gap is invisible, and the residues on both sides would be joined
    by one long peptide bond. We look for C(i)-N(i+1) distances above 2 Å.
    """
    pos = fixer.positions.value_in_unit(unit.angstrom)
    for ci, chain in enumerate(fixer.topology.chains()):
        residues = list(chain.residues())
        for k in range(len(residues) - 1):
            if (ci, k + 1) in fixer.missingResidues:
                continue  # A known gap that will be modeled.
            a, b = residues[k], residues[k + 1]
            c = next((x for x in a.atoms() if x.name == "C"), None)
            n = next((x for x in b.atoms() if x.name == "N"), None)
            if c is None or n is None:
                continue
            d = sum((pos[c.index][i] - pos[n.index][i]) ** 2 for i in range(3)) ** 0.5
            if d > 2.0:
                warnings.append(
                    f"Chain {chain.id} is broken between {a.name}{a.id} and {b.name}{b.id} "
                    f"(C-N distance {d:.1f} Å). The PDB file has no SEQRES information for this gap, "
                    "so it was not modeled, and the two residues would be joined by a stretched bond. "
                    "Fix the structure first, for example with an AlphaFold model.")


def build_system(
    pdb_path: str | Path,
    options: SetupOptions | None = None,
    target: list[tuple[str, int]] | None = None,
    remove_residues: list[tuple[str, int, str]] | None = None,
) -> BuiltSystem:
    opts = options or SetupOptions()
    opts.validate()
    preset = FORCEFIELDS[opts.forcefield]
    target_set = set(target or [])
    notes: list[str] = [f"Force field: {preset.description}."]
    warnings: list[str] = []

    fixer = PDBFixer(filename=str(pdb_path))
    if opts.chains:
        others = [c.id for c in fixer.topology.chains() if c.id not in opts.chains]
        fixer.removeChains(chainIds=others)
        notes.append(f"Kept chain(s) {', '.join(opts.chains)} only.")
    if remove_residues:
        _delete_residues(fixer, remove_residues)
        notes.append(f"Removed {len(remove_residues)} tag residue(s) at the chain ends.")
    removed = sorted({r.name for r in fixer.topology.residues()
                      if r.name not in _PROTEIN and r.name != "HOH"})
    fixer.removeHeterogens(keepWater=False)
    if removed:
        notes.append(f"Removed non-protein molecules: {', '.join(removed)}.")

    fixer.findMissingResidues()
    _gap_report(fixer, target_set, notes, warnings)
    _chain_break_check(fixer, warnings)
    if not fixer.missingResidues:
        notes.append("No missing residues inside the chains.")

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

    modeller = app.Modeller(fixer.topology, fixer.positions)
    hydrogens = [a for a in modeller.topology.atoms()
                 if a.element is not None and a.element.symbol == "H"]
    if hydrogens:
        modeller.delete(hydrogens)
        notes.append(f"Removed {len(hydrogens)} hydrogens from the input, to add them again consistently.")

    variants = _choose_protonation(modeller, opts, target_set, notes, warnings)

    try:
        forcefield = app.ForceField(*preset.files)
    except ValueError as e:
        raise ValueError(f"This OpenMM version does not include the {opts.forcefield} files "
                         f"({', '.join(preset.files)}). Update OpenMM or choose another --forcefield.") from e
    # Without a force field, OpenMM places hydrogens with standard geometry.
    # Energy minimization before MD relaxes them. This is much faster for CHARMM.
    used = modeller.addHydrogens(None, pH=opts.ph, variants=variants)
    his = [f"{r.name}{r.id}->{v}" for r, v in zip(modeller.topology.residues(), used)
           if r.name == "HIS" and v]
    notes.append(f"Added hydrogens at pH {opts.ph:g}." + (f" Histidines: {', '.join(his)}." if his else ""))

    # addSolvent uses a force field only to get the net charge and the atom sizes
    # for placing water. These are the same in all presets for standard residues.
    # Setting up CHARMM templates is slow, so we use Amber for this step.
    solvation_ff = forcefield if preset.solvent_geometry != "tip3p" else _amber14()
    solute = app.Modeller(modeller.topology, modeller.positions)
    padding = opts.padding_nm
    min_width = MIN_BOX_FACTOR * 2 * preset.cutoff_nm
    while True:
        modeller = app.Modeller(solute.topology, solute.positions)
        modeller.addSolvent(
            solvation_ff,
            model=preset.solvent_geometry,
            padding=padding * unit.nanometer,
            boxShape=opts.box_shape,
            ionicStrength=opts.ion_conc_molar * unit.molar,
            positiveIon=opts.cation,
            negativeIon="Cl-",
            neutralize=True,
        )
        width = _min_box_width_nm(modeller.topology.getPeriodicBoxVectors())
        if width >= min_width:
            break
        padding = round(padding + 0.2, 3)
    if padding != opts.padding_nm:
        notes.append(
            f"Increased the padding from {opts.padding_nm:g} to {padding:g} nm. With {opts.padding_nm:g} nm "
            f"the box would be narrower than {min_width:.2f} nm (twice the {preset.cutoff_nm:g} nm cutoff "
            "plus 10%), and pressure coupling could shrink it below the cutoff limit.")
    n_water = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
    salt = f"{opts.cation[:-1]}Cl"
    notes.append(
        f"Added a periodic water box ({opts.box_shape}, at least {padding:g} nm from the "
        f"protein to each face, {n_water} waters). Neutralized the charge and added "
        f"{salt} to {opts.ion_conc_molar:g} M.")

    kwargs = {}
    if preset.switch_nm is not None:
        kwargs["switchDistance"] = preset.switch_nm * unit.nanometer
    if opts.hydrogen_mass_amu:
        kwargs["hydrogenMass"] = opts.hydrogen_mass_amu * unit.amu
        notes.append(f"Hydrogen mass set to {opts.hydrogen_mass_amu:g} amu.")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=preset.cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
        **kwargs,
    )
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseDispersionCorrection(preset.dispersion_correction)
        elif isinstance(force, openmm.CustomNonbondedForce):
            force.setUseLongRangeCorrection(preset.dispersion_correction)
    switch = f", switched from {preset.switch_nm:g} nm" if preset.switch_nm else ""
    notes.append(
        f"Electrostatics: PME. Lennard-Jones cutoff {preset.cutoff_nm:g} nm{switch}. "
        f"Long-range dispersion correction {'on' if preset.dispersion_correction else 'off'}. "
        f"System has {system.getNumParticles()} atoms.")
    return BuiltSystem(modeller.topology, modeller.positions, system, notes, warnings)


def _choose_protonation(modeller, opts: SetupOptions, target, notes, warnings) -> list[str | None]:
    residues = list(modeller.topology.residues())
    variants: list[str | None] = [None] * len(residues)
    index = {(r.chain.id, int(r.id)): i for i, r in enumerate(residues)}
    # Force fields lack templates for most changed states at a chain end
    # (for example a neutral Glu as the first residue), so we leave chain ends alone.
    chain_ends = set()
    for chain in modeller.topology.chains():
        rs = list(chain.residues())
        chain_ends |= {rs[0].index, rs[-1].index}

    if opts.check_protonation:
        preds = protonation.predict(modeller.topology, modeller.positions, opts.ph)
        notes.append(protonation.format_predictions(preds, opts.ph, target))
        for p in preds:
            i = index.get((p.chain, p.resseq))
            if not p.changed or i is None:
                continue
            if i in chain_ends:
                notes.append(f"{p.resname}{p.resseq} chain {p.chain} is at a chain end, so it keeps "
                             "the default state. The force field has no template for the predicted state there.")
                continue
            variants[i] = p.variant

    for spec, state in opts.protonation.items():
        chain, _, num = spec.rpartition(":")
        matches = [i for (c, n), i in index.items() if n == int(num) and (not chain or c == chain)]
        if not matches:
            raise ValueError(f"Protonation choice {spec}={state}: no such residue.")
        for i in matches:
            name = residues[i].name
            if i in chain_ends:
                raise ValueError(f"Protonation choice {spec}={state}: {name}{residues[i].id} is at a "
                                 "chain end. The force field has no template for a changed state there.")
            if state not in protonation.ALLOWED.get(name, ()):
                allowed = protonation.ALLOWED.get(name)
                raise ValueError(f"Protonation choice {spec}={state}: residue is {name}. "
                                 + (f"Allowed: {', '.join(allowed)}." if allowed
                                    else "Its protonation state cannot be changed."))
            variants[i] = state
            notes.append(f"User choice: {name}{residues[i].id} chain {residues[i].chain.id} set to {state}.")
    return variants


def load_amber(prmtop: str | Path, inpcrd: str | Path) -> BuiltSystem:
    """Use a system the user prepared with Amber tools (for example tleap with GAFF for a ligand).

    Nothing is changed: no atoms are added or removed, and ligands keep their parameters.
    The file must contain a periodic box (a solvated system).
    """
    prm = app.AmberPrmtopFile(str(prmtop))
    crd = app.AmberInpcrdFile(str(inpcrd))
    if crd.boxVectors is None:
        raise ValueError(f"{inpcrd} has no periodic box. Solvate the system first (for example with tleap).")
    prm.topology.setPeriodicBoxVectors(crd.boxVectors)
    system = prm.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )
    names: dict[str, int] = {}
    prot = set(protein_atom_indices(prm.topology))
    for res in prm.topology.residues():
        if res.name != "WAT" and res.name != "HOH" and not any(a.index in prot for a in res.atoms()):
            names[res.name] = names.get(res.name, 0) + 1
    notes = [
        f"Using the prepared Amber system {Path(prmtop).name} / {Path(inpcrd).name} as it is.",
        f"System has {system.getNumParticles()} atoms. Lennard-Jones cutoff 1 nm, PME electrostatics.",
        "Non-protein residues: " + (", ".join(f"{k} x{v}" for k, v in sorted(names.items())) or "none") + ".",
    ]
    return BuiltSystem(prm.topology, crd.positions, system, notes, [])
