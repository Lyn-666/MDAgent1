"""Backbone conformational basins: Ramachandran regions of one residue, and the type of a beta turn.

Two ways to define basins (both written down before any data exist):
- regions ("boxes", the default for rama since the 2026-11-22 merge, user's choice): the phi/psi map is divided
  into regions that cover all of it (backbone.DEFAULT_BASINS_TEXT: alpha_R, beta with PPII, alpha_L, other),
  and a frame is in a region only `margin_deg` inside its edges. Every frame belongs to some region's core or
  to a gap between cores, so glycine's region (phi near 80, psi near 180) has its own basin. Users can give
  their own regions (--basins).
- centres: a few named basins, each a centre in angle space. A frame is in a basin when it lies within
  `core_deg` of the basin's centre (the distance is measured on the torus, so -179 and +179 degrees are 2
  degrees apart). Used for beta turns, and for rama with --rama-basins centres (runs made before the merge).

Frames outside
every core keep the basin they were last in, as for the chi1 states (diagnose.assign_states).
This "transition-based" assignment counts a transition only when the next core is reached, so
recrossings of a boundary are not counted as transitions; that matters for rates.

Two kinds are predefined, both written down before any data exist:

- "rama:A:73"  the phi/psi of residue 73: by default the regions alpha_R, beta (including polyproline
               II), alpha_L and other (see above); with centres, alphaR, beta and alphaL.
- "turn:A:52"  the turn whose second residue (i+1) is residue 52: psi of residue 52 and phi of
               residue 53 (where the centres come from: TURN_CENTRES). The basin centres are the ideal angles of
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
    boxes: list | None = None    # Regions (backbone.Basin) instead of centres; used when given.
    margin_deg: float = 10.0     # Regions: a frame must be this far inside a region to be in its core.

    @property
    def label(self) -> str:
        where = (f"{self.chain}:" if self.chain else "") + str(self.resseq)
        return f"{self.kind}:{where}"

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.boxes] if self.boxes else list(self.centres)

    def residues(self) -> list[tuple]:
        """Residues whose backbone atoms are needed (neighbours included, for phi and psi)."""
        nums = sorted({self.resseq + off + d for off, _ in self.dihedrals for d in (-1, 0, 1)})
        return [(self.chain, n) for n in nums]

    @classmethod
    def parse(cls, text: str, rama: str = "boxes", boxes_text: str = "", margin_deg: float = 10.0) -> "BasinSet":
        """rama: "boxes" (regions covering the map; boxes_text replaces the default regions) or "centres"."""
        m = _SPEC.match(text)
        if not m:
            raise ValueError(f"cannot read the backbone observable {text!r}. Write rama:A:73 (the phi/psi "
                             "regions of residue 73) or turn:A:52 (the beta turn whose residue i+1 is 52).")
        kind, chain, resseq = m.group(1), m.group(2), int(m.group(3))
        if kind == "rama":
            if rama == "centres":
                return cls(kind, chain, resseq, [(0, "phi"), (0, "psi")], dict(RAMA_CENTRES))
            if rama != "boxes":
                raise ValueError(f"unknown kind of Ramachandran basins {rama!r}: use boxes or centres.")
            from .backbone import parse_basins
            return cls(kind, chain, resseq, [(0, "phi"), (0, "psi")], {}, boxes=parse_basins(boxes_text),
                       margin_deg=margin_deg)
        return cls(kind, chain, resseq, [(0, "psi"), (1, "phi")], dict(TURN_CENTRES))


_RANGE = re.compile(r"^\s*(rama|turn)\s*:\s*(?:([A-Za-z0-9])\s*:\s*)?(-?\d+)\s*-\s*(-?\d+)\s*$")


def parse_sets(text: str | list | None, rama: str = "boxes", boxes_text: str = "",
               margin_deg: float = 10.0) -> list[BasinSet]:
    """Several observables separated by ';' or ',' (or a list). A range such as rama:A:2-75 expands.

    rama, boxes_text, margin_deg: how Ramachandran basins are defined (see BasinSet.parse)."""
    if not text:
        return []
    parts = text if isinstance(text, list) else [p for p in re.split(r"[;,]", text) if p.strip()]
    kw = dict(rama=rama, boxes_text=boxes_text, margin_deg=margin_deg)
    out = []
    for part in parts:
        m = _RANGE.match(part)
        if m:
            kind, chain, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            out += [BasinSet.parse(f"{kind}:{chain + ':' if chain else ''}{n}", **kw) for n in range(a, b + 1)]
        else:
            out.append(BasinSet.parse(part, **kw))
    return out


def residue_sets(target: list, kind: str = "rama") -> str:
    """The spec text for every residue of a target list, for example [("A", 110)] -> "rama:A:110"."""
    return ";".join(f"{kind}:{c}:{n}" if c else f"{kind}:{n}" for c, n in target)


def torus_distance(values: np.ndarray, centre) -> np.ndarray:
    """Distance (degrees) between points and a centre on the torus of angles."""
    d = np.abs((np.asarray(values, dtype=float) - np.asarray(centre, dtype=float) + 180.0) % 360.0 - 180.0)
    return np.sqrt(np.sum(np.atleast_2d(d) ** 2, axis=-1)) if np.ndim(values) > 1 else d


def core(values: np.ndarray, basins: BasinSet) -> np.ndarray:
    """Index (order of basins.names) of the basin core each point is in; -1 between cores."""
    v = np.asarray(values, dtype=float).reshape(len(values), -1)
    if basins.boxes:
        from .backbone import core_index
        return core_index(v[:, 0], v[:, 1], basins.boxes, basins.margin_deg)
    out = np.full(len(v), -1, dtype=int)
    best = np.full(len(v), np.inf)
    for i, c in enumerate(basins.centres.values()):
        d = torus_distance(v, c)
        hit = (d <= basins.core_deg) & (d < best)
        out[hit], best[hit] = i, d[hit]
    return out


def assign(values: np.ndarray, basins: BasinSet) -> np.ndarray:
    """State index of each frame (order of basins.names); -1 before the first core is reached. Between cores a
    frame keeps the last basin (transition-based assignment)."""
    states = core(values, basins).copy()
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


# ---------------------------------------------------------------------------------------------
# Everything one item tells about itself, from frame-by-frame states (and angles, for the landscape).
# Used by the live agent and by the offline benchmark, so both judge the same numbers.

def histogram(values: list, bins: int = 36) -> np.ndarray:
    """Normalised histogram of the angles of all replicas (1D or 2D, 10 degree bins by default)."""
    v = [np.asarray(x, float).reshape(len(x), -1) for x in values if len(x)]
    if not v:
        return np.zeros(bins)
    v = np.concatenate(v)
    if v.shape[1] == 1:
        h, _ = np.histogram(v[:, 0], bins=bins, range=(-180, 180))
    else:
        h, _, _ = np.histogram2d(v[:, 0], v[:, 1], bins=bins, range=[[-180, 180], [-180, 180]])
    h = h.astype(float).ravel()
    return h / h.sum() if h.sum() else h


def js_bits(p: np.ndarray, q: np.ndarray, support: np.ndarray | None = None) -> float:
    """Jensen-Shannon divergence in bits (0 to 1), optionally only on the given bins."""
    if support is not None:
        p, q = p[support], q[support]
    if p.sum() <= 0 or q.sum() <= 0:
        return 1.0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))
    return 0.5 * kl(p) + 0.5 * kl(q)


def summarize(states: list, values: list | None, names: list, ns_per_frame: float) -> dict:
    """Populations with errors (time correlation and replica spread), the free-energy difference of the
    first two basins (kT), the rates between them, and the landscape histogram with its split-half check.

    states: one array of state indices per replica (-1 before the first core). values: the angles,
    one (frames, dims) array per replica, or None (then no landscape).
    """
    import math

    from .uncertainty import block_standard_error, error_parts

    pops, se, within, between = {}, {}, {}, {}
    for k, name in enumerate(names):
        hits = [(s[s >= 0] == k).astype(float) for s in states]
        vals = [float(h.mean()) if h.size else 0.0 for h in hits]
        errs = [block_standard_error(h) for h in hits]
        w, bt = error_parts(vals, errs)
        pops[name] = float(np.mean(vals))
        se[name] = max(w, bt) if len(vals) > 1 else (errs[0] if np.isfinite(errs[0]) else float("nan"))
        within[name], between[name] = w, bt
    trans, counts, dwell = [], np.zeros((len(names), len(names))), np.zeros(len(names))
    for s in states:
        s = s[s >= 0]
        change = np.nonzero(s[1:] != s[:-1])[0]
        trans.append(int(len(change)))
        for i in change:
            counts[s[i], s[i + 1]] += 1
        for k in range(len(names)):
            dwell[k] += np.sum(s == k) * ns_per_frame
    a, c = 0, 1
    pa, pc = pops[names[a]], pops[names[c]]
    dg = -math.log(pc / pa) if pa > 0 and pc > 0 else float("inf")
    # Conservative: the two errors add, as if fully correlated (two-state populations are).
    dg_se = (se[names[a]] / pa + se[names[c]] / pc) if pa > 0 and pc > 0 else float("inf")
    rates = {}
    for i, j in ((a, c), (c, a)):
        nij = counts[i, j]
        rates[f"{names[i]}->{names[j]}"] = {"k_per_ns": float(nij / dwell[i]) if dwell[i] > 0 else 0.0,
                                            "n": int(nij), "rel_se": 1 / math.sqrt(nij) if nij else float("inf")}
    out = {"populations": pops, "se": se, "transitions": trans, "dg_kt": dg, "dg_se": dg_se, "rates": rates}
    worst = max(names, key=lambda k: se[k] if np.isfinite(se[k]) else 1.0)
    out["within"], out["between"] = within[worst], between[worst]
    if values is not None:
        out["hist"] = histogram(values)
        halves = ([values[i] for i in range(0, len(values), 2)], [values[i] for i in range(1, len(values), 2)]) \
            if len(values) > 1 else ([values[0][: len(values[0]) // 2]], [values[0][len(values[0]) // 2:]])
        out["split_js"] = js_bits(histogram(halves[0]), histogram(halves[1]))
    return out


def judge(est: dict, z: float, tolerance: float, min_transitions: int = 4, dg_kt: float = 0.5,
          rate_rel: float = 0.2, landscape_js: float = 0.05) -> dict:
    """The method's own verdict on each observable, from its own data only (never a reference)."""
    adequate = bool(est["transitions"]) and min(est["transitions"]) >= min_transitions
    pops_ok = all(np.isfinite(v) and z * v <= tolerance for v in est["se"].values())
    rates_ok = all(r["n"] > 0 and z * r["rel_se"] <= rate_rel for r in est["rates"].values())
    return {"populations": pops_ok and adequate, "adequate": adequate,
            "dg": adequate and np.isfinite(est["dg_se"]) and z * est["dg_se"] <= dg_kt,
            "rates": adequate and rates_ok,
            "landscape": adequate and est.get("split_js", 1.0) <= landscape_js}
