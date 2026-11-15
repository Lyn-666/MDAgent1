"""The target: the region of the protein that the user cares about.

A target is either an explicit list of residues, or all protein residues
within a cutoff distance of a ligand in the input PDB file.
Residues are identified by (chain ID, residue number) as written in the PDB file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import mdtraj as md

ResidueKey = tuple[str, int]  # (chain ID, residue number)


@dataclass(frozen=True)
class Target:
    residues: tuple[tuple[str | None, int], ...] = field(default_factory=tuple)
    ligand: str | None = None
    cutoff_nm: float = 0.5

    def describe(self) -> str:
        if self.ligand:
            return f"protein residues within {self.cutoff_nm * 10:.1f} Å of ligand {self.ligand}"
        return "residues " + ", ".join(
            f"{c}:{n}" if c else str(n) for c, n in self.residues
        )


_SPEC_ITEM = re.compile(r"^(?:(?P<chain>[A-Za-z0-9]):)?(?P<start>-?\d+)(?:-(?P<end>-?\d+))?$")


def parse_residue_spec(spec: str) -> tuple[tuple[str | None, int], ...]:
    """Parse a residue list such as "A:99-120,A:133" or "99-120".

    A missing chain ID means "any chain".
    """
    out: list[tuple[str | None, int]] = []
    for item in spec.replace(" ", "").split(","):
        if not item:
            continue
        m = _SPEC_ITEM.match(item)
        if not m:
            raise ValueError(f"Cannot parse residue item {item!r}. Use forms like 'A:99-120' or '133'.")
        start = int(m["start"])
        end = int(m["end"]) if m["end"] is not None else start
        if end < start:
            raise ValueError(f"Residue range {item!r} ends before it starts.")
        out.extend((m["chain"], n) for n in range(start, end + 1))
    if not out:
        raise ValueError("The residue list is empty.")
    return tuple(out)


def resolve_target(target: Target, pdb_path: str | Path) -> list[ResidueKey]:
    """Return the protein residues of the target, as found in the input PDB file."""
    traj = md.load_pdb(str(pdb_path))
    top = traj.topology
    protein = {r for r in top.residues if r.is_protein}

    if target.ligand:
        lig_atoms = [a.index for a in top.atoms if a.residue.name == target.ligand]
        if not lig_atoms:
            names = sorted({r.name for r in top.residues if not r.is_protein and not r.is_water})
            raise ValueError(f"Ligand {target.ligand!r} is not in the PDB file. Found: {names}")
        prot_atoms = [a.index for a in top.atoms if a.residue in protein]
        near = md.compute_neighbors(traj, target.cutoff_nm, lig_atoms, haystack_indices=prot_atoms)[0]
        found = {top.atom(int(i)).residue for i in near}
    else:
        wanted = set(target.residues)
        found = {
            r for r in protein
            if (r.chain.chain_id, r.resSeq) in wanted or (None, r.resSeq) in wanted
        }

    keys = sorted({(r.chain.chain_id, r.resSeq) for r in found})
    if not keys:
        raise ValueError(f"No protein residues match the target: {target.describe()}.")
    return keys


def pdb_summary(pdb_path: str | Path) -> str:
    """Chains, residue ranges and non-protein molecules of a PDB or Amber topology (.parm7/.prmtop) file."""
    if Path(pdb_path).suffix.lower() in (".parm7", ".prmtop"):
        # Read Amber files through OpenMM, as `mdagent run --prmtop` does, so that residue
        # numbers match (OpenMM numbers them from 1; MDTraj's own reader starts at 0).
        from openmm import app

        top = md.Topology.from_openmm(app.AmberPrmtopFile(str(pdb_path)).topology)
    else:
        top = md.load_topology(str(pdb_path))
    lines = []
    for chain in top.chains:
        prot = [r for r in chain.residues if r.is_protein]
        if prot:
            name = chain.chain_id or f"#{chain.index + 1} (no chain ID in the file)"
            lines.append(f"Chain {name}: protein residues {prot[0].resSeq}-{prot[-1].resSeq} "
                         f"({len(prot)} residues)")
    others: dict[str, int] = {}
    for r in top.residues:
        if not r.is_protein and not r.is_water:
            others[r.name] = others.get(r.name, 0) + 1
    if others:
        lines.append("Non-protein molecules (residue name x count): "
                     + ", ".join(f"{k} x{v}" for k, v in sorted(others.items())))
    else:
        lines.append("No ligands or other non-protein molecules.")
    return "\n".join(lines)
