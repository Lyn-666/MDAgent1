"""Build a peptide from its sequence, so that a run can start without a PDB file.

The user writes the sequence in one-letter code, optionally with the end groups, for example
"Ac-AAKAAAAKAA-NH2" (acetyl at the N end, amide at the C end) or "GSWTWENGKWTWK" (free, charged ends;
"H-...-OH" means the same). Only the 20 standard amino acids are supported: lowercase letters (D-amino acids),
modified residues and cyclic peptides need parameters MDAgent does not have.

The backbone atoms (N, CA, C, O) are placed with standard bond lengths and angles (Engh and Huber) in an
extended (phi -135, psi 135) or helical (phi -57, psi -47) conformation; proline gets phi -65. Side chains and
hydrogens are added by the normal system build (PDBFixer adds missing atoms from its templates), and Ac / NH2
become the ACE / NME caps of build.py. Extended is the default start: it does not favour a helix, and the
automated burn-in leaves out the relaxation from it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ONE_TO_THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
                "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
                "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
CONFORMATIONS = {"extended": (-135.0, 135.0), "helix": (-57.0, -47.0)}

# Backbone geometry (nm, degrees).
N_CA, CA_C, C_N, C_O = 0.1458, 0.1525, 0.1329, 0.1231
ANG_N_CA_C, ANG_CA_C_N, ANG_C_N_CA, ANG_CA_C_O = 111.2, 116.2, 121.7, 120.5


@dataclass
class Peptide:
    residues: list          # three-letter names
    n_cap: bool             # Ac- at the N end (ACE)
    c_cap: bool             # -NH2 at the C end (NME)
    text: str               # the sequence as the user wrote it, normalised

    @property
    def one_letter(self) -> str:
        three_to_one = {v: k for k, v in ONE_TO_THREE.items()}
        return "".join(three_to_one[r] for r in self.residues)


def parse_sequence(text: str) -> Peptide:
    """Read "Ac-AAKAA-NH2", "H-AAKAA-OH" or "AAKAA". Raises ValueError with a plain reason."""
    s = re.sub(r"\s+", "", text or "")
    n_cap = c_cap = False
    head = re.match(r"^(Ac|H)-", s, flags=re.IGNORECASE)
    if head:
        n_cap = head.group(1).lower() == "ac"
        s = s[head.end():]
    tail = re.search(r"-(NH2|OH)$", s, flags=re.IGNORECASE)
    if tail:
        c_cap = tail.group(1).upper() == "NH2"
        s = s[:tail.start()]
    if not s:
        raise ValueError("the peptide sequence is empty. Write it in one-letter code, for example Ac-AAKAA-NH2.")
    bad = sorted({ch for ch in s if ch not in ONE_TO_THREE})
    if bad:
        lower = [ch for ch in bad if ch.isalpha() and ch.islower()]
        why = (" Lowercase letters (D-amino acids) are not supported." if lower else "")
        raise ValueError(f"the sequence has letters that are not standard amino acids: {', '.join(bad)}.{why} "
                         "Only the 20 standard amino acids in one-letter code are supported; modified residues and "
                         "cyclic peptides need a prepared Amber system (--prmtop).")
    if len(s) < 2:
        raise ValueError("a peptide needs at least 2 residues.")
    if len(s) > 200:
        raise ValueError("this sequence is longer than 200 residues; for a protein, start from a PDB file instead.")
    norm = ("Ac-" if n_cap else "") + s + ("-NH2" if c_cap else "")
    return Peptide([ONE_TO_THREE[ch] for ch in s], n_cap, c_cap, norm)


def _place(a, b, c, bond, angle_deg, torsion_deg):
    """The point d with |cd| = bond, angle bcd = angle and dihedral abcd = torsion (NeRF, Parsons et al. 2005)."""
    angle, torsion = math.radians(angle_deg), math.radians(torsion_deg)
    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-bond * math.cos(angle), bond * math.sin(angle) * math.cos(torsion),
                   bond * math.sin(angle) * math.sin(torsion)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def backbone(peptide: Peptide, conformation: str = "extended") -> list[dict]:
    """Backbone atoms (N, CA, C, O) of every residue: [{"name", "resname", "resseq", "xyz" (nm)}]."""
    if conformation not in CONFORMATIONS:
        raise ValueError(f"unknown start conformation {conformation!r}; choose from {list(CONFORMATIONS)}.")
    phi0, psi0 = CONFORMATIONS[conformation]
    atoms = []
    n = np.array([0.0, 0.0, 0.0])
    ca = np.array([N_CA, 0.0, 0.0])
    # A first C in the plane, at the N-CA-C angle.
    ang = math.radians(180.0 - ANG_N_CA_C)
    c = ca + CA_C * np.array([math.cos(ang), math.sin(ang), 0.0])
    for i, res in enumerate(peptide.residues):
        phi = -65.0 if res == "PRO" else phi0
        if i > 0:
            n = _place(prev_n, prev_ca, prev_c, C_N, ANG_CA_C_N, psi_prev)
            ca = _place(prev_ca, prev_c, n, N_CA, ANG_C_N_CA, 180.0)          # omega, trans
            c = _place(prev_c, n, ca, CA_C, ANG_N_CA_C, phi)
        o = _place(n, ca, c, C_O, ANG_CA_C_O, psi0 + 180.0)
        for name, xyz in (("N", n), ("CA", ca), ("C", c), ("O", o)):
            atoms.append({"name": name, "resname": res, "resseq": i + 1, "xyz": xyz.copy()})
        prev_n, prev_ca, prev_c, psi_prev = n, ca, c, psi0
    return atoms


def write_pdb(peptide: Peptide, path: Path, conformation: str = "extended") -> Path:
    """Write the backbone as chain A, residues 1..n (side chains and hydrogens come in the system build)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"REMARK   MDAgent peptide {peptide.text}, start conformation {conformation}"]
    for k, a in enumerate(backbone(peptide, conformation), start=1):
        x, y, z = (a["xyz"] * 10.0).tolist()     # nm to Angstrom
        lines.append(f"ATOM  {k:5d}  {a['name']:<3s} {a['resname']:>3s} A{a['resseq']:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {a['name'][0]:>1s}")
    lines += ["TER", "END"]
    path.write_text("\n".join(lines) + "\n")
    return path


def default_path(peptide: Peptide, conformation: str, root: Path | None = None) -> Path:
    """structures/peptide_<sequence>_<conformation>.pdb (clear to the user at a glance)."""
    safe = re.sub(r"[^A-Za-z0-9-]", "", peptide.text)
    return Path(root or Path.cwd()) / "structures" / f"peptide_{safe}_{conformation}.pdb"
