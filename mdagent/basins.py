"""Backbone conformational basins: Ramachandran regions of one residue, and the type of a beta turn.

A basin set is one or two backbone dihedrals and a few named basins, each a centre in that
angle space. A frame is in a basin when it lies within `core_deg` of the basin's centre (the
distance is measured on the torus, so -179 and +179 degrees are 2 degrees apart). Frames outside
every core keep the basin they were last in, as for the chi1 states (diagnose.assign_states).
This "transition-based" assignment counts a transition only when the next core is reached, so
recrossings of a boundary are not counted as transitions; that matters for rates.

Two kinds are predefined, both written down before any data exist:

- "rama:A:73"  the phi/psi of residue 73, with the conventional regions alphaR, beta (including
               polyproline II) and alphaL.
- "turn:A:52"  the turn whose second residue (i+1) is residue 52: psi of residue 52 and phi of
               residue 53. For ubiquitin this is the D52/G53 peptide of the 50s loop, which flips
               between a type I and a type II beta turn. The basin centres are the ideal angles of
               the two turn types (TURN_CENTRES); a pilot or the literature can refine them, but only
               before the pool is analysed.

The raw angles are kept with the states, so that a free-energy landscape (a histogram of the
angles) can be compared between runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Conventional Ramachandran regions (degrees, phi then psi). beta is centred so that it also
# covers polyproline II (about -75, 145).
RAMA_CENTRES = {"alphaR": (-63.0, -43.0), "beta": (-100.0, 140.0), "alphaL": (57.0, 47.0)}
# Turn angles, in the order (psi of residue i+1, phi of residue i+2). Type I: psi(i+1) -30, phi(i+2) -90
# (the classical type I turn). Type II: psi(i+1) 150, phi(i+2) 90, as used for the ubiquitin D52/G53
# flip by Arasteh, Zhang & Levy 2021 (doi:10.1021/acs.jpclett.1c00778); the observed type II turn with
# Gly at i+2 is close to it (133, 91; Shapovalov, Vucetic & Dunbrack 2019, doi:10.1371/journal.pcbi.1006844).
TURN_CENTRES = {"betaI": (-30.0, -90.0), "betaII": (150.0, 90.0)}
# Radius of a basin core. Turn assignments usually allow about 30 degrees on each angle and 45 on one
# (de Brevern 2016, doi:10.1038/srep33191); 50 degrees in the two-angle plane is comparable.
CORE_DEG = 50.0

_SPEC = re.compile(r"^\s*(rama|turn)\s*:\s*(?:([A-Za-z0-9])\s*:\s*)?(-?\d+)\s*$")


@dataclass
class BasinSet:
    kind: str                       # "rama" or "turn"
    chain: str | None
    resseq: int
    # (residue offset from resseq, "phi" or "psi") for each dimension, in order.
    dihedrals: list = field(default_factory=list)
    centres: dict = field(default_factory=dict)     # basin name -> centre (degrees), one value per dimension
    core_deg: float = CORE_DEG

    @property
    def label(self) -> str:
        where = (f"{self.chain}:" if self.chain else "") + str(self.resseq)
        return f"{self.kind}:{where}"

    @property
    def names(self) -> list[str]:
        return list(self.centres)

    def residues(self) -> list[tuple]:
        """Residues whose backbone atoms are needed (neighbours included, for phi and psi)."""
        nums = sorted({self.resseq + off + d for off, _ in self.dihedrals for d in (-1, 0, 1)})
        return [(self.chain, n) for n in nums]

    @classmethod
    def parse(cls, text: str) -> "BasinSet":
        m = _SPEC.match(text)
        if not m:
            raise ValueError(f"cannot read the backbone observable {text!r}. Write rama:A:73 (the phi/psi "
                             "regions of residue 73) or turn:A:52 (the beta turn whose residue i+1 is 52).")
        kind, chain, resseq = m.group(1), m.group(2), int(m.group(3))
        if kind == "rama":
            return cls(kind, chain, resseq, [(0, "phi"), (0, "psi")], dict(RAMA_CENTRES))
        return cls(kind, chain, resseq, [(0, "psi"), (1, "phi")], dict(TURN_CENTRES))


_RANGE = re.compile(r"^\s*(rama|turn)\s*:\s*(?:([A-Za-z0-9])\s*:\s*)?(-?\d+)\s*-\s*(-?\d+)\s*$")


def parse_sets(text: str | list | None) -> list[BasinSet]:
    """Several observables separated by ';' or ',' (or a list). A range such as rama:A:2-75 expands."""
    if not text:
        return []
    parts = text if isinstance(text, list) else [p for p in re.split(r"[;,]", text) if p.strip()]
    out = []
    for part in parts:
        m = _RANGE.match(part)
        if m:
            kind, chain, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            out += [BasinSet.parse(f"{kind}:{chain + ':' if chain else ''}{n}") for n in range(a, b + 1)]
        else:
            out.append(BasinSet.parse(part))
    return out


def torus_distance(values: np.ndarray, centre) -> np.ndarray:
    """Distance (degrees) between points and a centre on the torus of angles."""
    d = np.abs((np.asarray(values, dtype=float) - np.asarray(centre, dtype=float) + 180.0) % 360.0 - 180.0)
    return np.sqrt(np.sum(np.atleast_2d(d) ** 2, axis=-1)) if np.ndim(values) > 1 else d


def assign(values: np.ndarray, basins: BasinSet) -> np.ndarray:
    """State index of each frame (order of basins.centres); -1 before the first core is reached."""
    v = np.asarray(values, dtype=float).reshape(len(values), -1)
    core = np.full(len(v), -1, dtype=int)
    best = np.full(len(v), np.inf)
    for i, c in enumerate(basins.centres.values()):
        d = torus_distance(v, c)
        hit = (d <= basins.core_deg) & (d < best)
        core[hit], best[hit] = i, d[hit]
    states = core.copy()
    for t in range(1, len(states)):
        if states[t] == -1:
            states[t] = states[t - 1]
    return states


def dihedral_atoms(top, basins: BasinSet) -> list[list[int]]:
    """Atom indices (in `top`) of each dihedral of the set. Raises ValueError for missing atoms."""
    out = []
    for off, angle in basins.dihedrals:
        n = basins.resseq + off
        chain = basins.chain

        def atom(resseq: int, name: str) -> int:
            hits = [a.index for a in top.atoms if a.residue.resSeq == resseq and a.name == name
                    and (chain is None or a.residue.chain.chain_id == chain)]
            if len(hits) != 1:
                raise ValueError(f"{basins.label}: atom {name} of residue {resseq} "
                                 f"{'is missing' if not hits else 'is not unique'}.")
            return hits[0]
        if angle == "phi":
            out.append([atom(n - 1, "C"), atom(n, "N"), atom(n, "CA"), atom(n, "C")])
        else:
            out.append([atom(n, "N"), atom(n, "CA"), atom(n, "C"), atom(n + 1, "N")])
    return out


def angles(traj, quads: list[list[int]]) -> np.ndarray:
    """(frames, dimensions) dihedral angles in degrees."""
    import mdtraj as md

    return np.degrees(md.compute_dihedrals(traj, np.asarray(quads)))
