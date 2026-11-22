"""Backbone phi/psi of any residue: Ramachandran basins, transition statistics, free-energy landscape.

Nothing here is tied to a protein, a residue or a test system. Three results per residue:

1. Basin populations. Each frame is put into a basin of the Ramachandran map. A frame counts as in a
   basin only inside its core (the basin shrunk by `margin` degrees on every side); between cores it
   keeps the last basin, as in core-set Markov models (Buchete and Hummer, J. Phys. Chem. B 2008, 112,
   6057). The basins are data, not code: the user can give their own (parse_basins). The default divides
   the map into the conventional regions (for example Hovmoller, Zhou and Ohlson, Acta Cryst. D 2002, 58,
   768): alpha_R (phi < 0, -120 <= psi < 50), beta (phi < 0, the other psi; beta and PPII together),
   alpha_L (phi >= 0, -60 <= psi < 100) and other (phi >= 0, the other psi; mainly glycine). Populations
   and errors use the same code as every other quantity (diagnose.populations_with_errors).
2. Transition statistics between basins: counts, rates k_ij = N_ij / (time in i) with Poisson errors
   sqrt(N_ij) / (time in i), and mean dwell times (time in i / number of exits from i, the maximum-likelihood
   estimate for Markovian jumps; it also uses the visits cut off at the start and the end of a trajectory, so
   long visits are not left out) with standard error dwell / sqrt(exits). These errors assume Markovian jumps
   between cores; they are too small if visits are correlated.
3. The phi/psi free-energy landscape F = -kT ln p on a grid, shifted so that its minimum is 0. The error
   of p in each bin is the spread between replicas (never below an error floor from the number of basin
   transitions, as for populations), propagated to F as kT * se(p) / p.

Each result has its own success criterion (agent.Goal): the population error (precision), the largest
relative error of the leading rates (rate_rel_error), and the largest error of F where F <= fes_cutoff_kcal
(fes_error_kcal). There is no shared threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .uncertainty import floor_from_samples

KB_KCAL = 0.0019872043   # kcal/(mol K)


@dataclass(frozen=True)
class Basin:
    name: str
    boxes: tuple   # ((phi_lo, phi_hi, psi_lo, psi_hi), ...) in degrees; lo > hi wraps through 180.


DEFAULT_BASINS_TEXT = ("alpha_R:-180,0,-120,50; beta:-180,0,50,-120; alpha_L:0,180,-60,100; "
                       "other:0,180,100,-60")


def _norm(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def _inside(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """x in [lo, hi) on the circle. lo == hi after normalizing means the full circle only if the width is 360."""
    lo, hi = float(_norm(lo)), float(_norm(hi))
    if lo < hi:
        return (x >= lo) & (x < hi)
    return (x >= lo) | (x < hi)


def _width(lo: float, hi: float) -> float:
    w = (hi - lo) % 360.0
    return 360.0 if w == 0 else w


def parse_basins(text: str | None) -> list[Basin]:
    """"name:phi_lo,phi_hi,psi_lo,psi_hi; ..." (degrees). A name may repeat to add boxes to that basin."""
    text = text or DEFAULT_BASINS_TEXT
    boxes: dict[str, list] = {}
    for part in [p for p in text.split(";") if p.strip()]:
        name, sep, nums = part.partition(":")
        name = name.strip()
        try:
            values = tuple(float(v) for v in nums.split(","))
        except ValueError:
            values = ()
        if not sep or not name or len(values) != 4:
            raise ValueError(f"cannot read the basin {part.strip()!r}. Write it like alpha_R:-180,0,-120,50 "
                             "(name:phi_from,phi_to,psi_from,psi_to in degrees).")
        if not all(-360 <= v <= 360 for v in values):
            raise ValueError(f"the angles of basin {name!r} must be in degrees between -180 and 180.")
        boxes.setdefault(name, []).append(values)
    if len(boxes) < 2:
        raise ValueError("give at least two basins.")
    return [Basin(n, tuple(b)) for n, b in boxes.items()]


def basins_text(basins: list[Basin]) -> str:
    return "; ".join(f"{b.name}:{','.join(f'{v:g}' for v in box)}" for b in basins for box in b.boxes)


def _box_core(phi: np.ndarray, psi: np.ndarray, box: tuple, margin_deg: float) -> np.ndarray:
    """Inside the box shrunk by the margin on every side (at most a quarter of its width)."""
    hit = np.ones(phi.shape, dtype=bool)
    for x, lo, hi in ((phi, box[0], box[1]), (psi, box[2], box[3])):
        w = _width(lo, hi)
        if w < 360:
            m = min(margin_deg, w / 4)
            hit &= _inside(x, lo + m, hi - m)
    return hit


def core_index(phi_deg, psi_deg, basins: list[Basin], margin_deg: float = 10.0) -> np.ndarray:
    """Basin index of every point inside a basin core, -1 between cores. The first basin listed wins an overlap."""
    phi, psi = _norm(phi_deg), _norm(psi_deg)
    core = np.full(phi.shape, -1, dtype=int)
    for i, b in enumerate(basins):
        hit = np.zeros(phi.shape, dtype=bool)
        for box in b.boxes:
            hit |= _box_core(phi, psi, box, margin_deg)
        core[hit & (core < 0)] = i
    return core


def assign_basins(phi_deg, psi_deg, basins: list[Basin], margin_deg: float = 10.0) -> np.ndarray:
    """Basin index of every frame (core-set rule): inside a core, that basin; between cores, the last basin;
    before the first core, -1."""
    states = core_index(phi_deg, psi_deg, basins, margin_deg)
    for t in range(1, len(states)):
        if states[t] < 0:
            states[t] = states[t - 1]
    return states


# ---------------------------------------------------------------- which atoms

def backbone_quads(top, target: list) -> tuple[dict, list]:
    """phi and psi atom quadruples (indices in `top`) of the target residues.

    Returns ({label: ((chain, number), resname, phi_quad, psi_quad)}, [(key, reason), ...] left out).
    A residue needs both neighbors on its chain: the first residue has no phi, the last no psi.
    """
    import mdtraj as md

    t = md.Trajectory(np.zeros((1, top.n_atoms, 3)), top)
    phi_idx, _ = md.compute_phi(t)
    psi_idx, _ = md.compute_psi(t)
    phi = {(top.atom(int(q[2])).residue.chain.chain_id, top.atom(int(q[2])).residue.resSeq): tuple(map(int, q))
           for q in phi_idx}
    psi = {(top.atom(int(q[1])).residue.chain.chain_id, top.atom(int(q[1])).residue.resSeq): tuple(map(int, q))
           for q in psi_idx}
    names = {(r.chain.chain_id, r.resSeq): r.name for r in top.residues}
    found, skipped = [], []
    for key in target:
        key = tuple(key)
        if key not in names:
            skipped.append((key, "not in the structure"))
        elif key not in phi or key not in psi:
            skipped.append((key, "a chain end: it has no " + ("phi" if key not in phi else "psi")))
        else:
            found.append((f"{names[key]}{key[1]}", key))
    labels = [n for n, _ in found]
    out = {}
    for n, key in found:
        label = n if labels.count(n) == 1 else f"{key[0]}:{n}"
        out[label] = (key, names[key], phi[key], psi[key])
    return out, skipped


# ---------------------------------------------------------------- transition statistics

def transition_statistics(states: list, names: list, dt_ns: float) -> dict:
    """Counts, rates and dwell times between basins over all replicas (see the module notes)."""
    n = len(names)
    counts = np.zeros((n, n), dtype=int)
    frames = np.zeros(n)
    for s in states:
        s = np.asarray(s)
        s = s[s >= 0]
        if not len(s):
            continue
        frames += np.bincount(s, minlength=n)[:n]
        change = np.nonzero(s[1:] != s[:-1])[0]
        for k in change:
            counts[s[k], s[k + 1]] += 1
    time = frames * dt_ns
    rates, dwell_out = {}, {}
    for i, a in enumerate(names):
        rates[a] = {}
        for j, b in enumerate(names):
            if i != j and time[i] > 0:
                rates[a][b] = [counts[i, j] / time[i], math.sqrt(counts[i, j]) / time[i] if counts[i, j]
                               else 1.0 / time[i]]   # No event: the error is about one event.
        exits = int(counts[i].sum())
        dwell_out[a] = ([float(time[i] / exits), float(time[i] / exits / math.sqrt(exits)), exits] if exits >= 2
                        else None)
    return {"names": list(names), "dt_ns": dt_ns, "counts": counts.tolist(), "time_ns": time.round(6).tolist(),
            "rates_per_ns": rates, "mean_dwell_ns": dwell_out, "total_transitions": int(counts.sum())}


def worst_rate_error(stats: dict, populations: dict, min_population: float = 0.05) -> tuple[float, list]:
    """Largest relative rate error among the leading transitions, and those transitions.

    Leading: both directions between basins with population >= min_population. With only one such
    basin, the total rate of leaving it. A transition never seen has an infinite relative error.
    """
    names, counts = stats["names"], np.array(stats["counts"])
    lead = [i for i, a in enumerate(names) if populations.get(a, 0) >= min_population]
    if len(lead) >= 2:
        pairs = [(i, j) for i in lead for j in lead if i != j]
        events = [counts[i, j] for i, j in pairs]
        labels = [f"{names[i]}->{names[j]}" for i, j in pairs]
    else:
        top = lead[0] if lead else int(np.argmax(stats["time_ns"]))
        events = [int(counts[top].sum())]
        labels = [f"{names[top]}->any"]
    worst = max((1 / math.sqrt(e) if e else math.inf) for e in events)
    return worst, labels


# ---------------------------------------------------------------- free-energy landscape

def landscape(phi_list: list, psi_list: list, temperature_k: float, bins_deg: float = 15.0,
              n_effective: float | None = None, weights: list | None = None) -> dict:
    """F(phi, psi) in kcal/mol with errors, from one phi and one psi series per replica.

    n_effective: independent samples for the error floor (basin transitions + replicas); None: no floor.
    weights: per-frame weights per replica (reweighted biased sampling); replicas then count by their Kish
    effective sample size.
    """
    from .uncertainty import effective_samples

    nb = int(round(360.0 / bins_deg))
    edges = np.linspace(-180.0, 180.0, nb + 1)
    hists, sizes = [], []
    for k, (phi, psi) in enumerate(zip(phi_list, psi_list)):
        w = None if weights is None else np.asarray(weights[k], dtype=float)
        h, _, _ = np.histogram2d(_norm(phi), _norm(psi), bins=[edges, edges], weights=w)
        sizes.append(h.sum() if w is None else effective_samples(w))
        hists.append(h / h.sum() if h.sum() > 0 else h)
    weights_ = sizes
    w = np.array(weights_) / max(sum(weights_), 1e-300)
    p = np.tensordot(w, np.array(hists), axes=1)
    n = len(hists)
    se = (np.std(np.array(hists), axis=0, ddof=1) / math.sqrt(n)) if n > 1 else np.full(p.shape, np.nan)
    if n_effective:
        floor = np.vectorize(lambda x: floor_from_samples(x, n_effective) if x > 0 else 0.0)(p)
        se = np.fmax(np.nan_to_num(se, nan=0.0), floor)
    kt = KB_KCAL * temperature_k
    with np.errstate(divide="ignore", invalid="ignore"):
        f = -kt * np.log(p)
        f -= np.nanmin(f[np.isfinite(f)]) if np.isfinite(f).any() else 0.0
        err = kt * se / p
    f[~np.isfinite(f)] = np.nan
    err[~np.isfinite(f)] = np.nan
    return {"edges_deg": edges, "free_energy_kcal": f, "error_kcal": err, "probability": p, "n_replicas": n}


def worst_landscape_error(fes: dict, cutoff_kcal: float = 2.0) -> float:
    """Largest error of F among the bins with F <= cutoff (the populated region). inf without replicas to compare."""
    f, e = fes["free_energy_kcal"], fes["error_kcal"]
    mask = np.isfinite(f) & (f <= cutoff_kcal)
    if fes["n_replicas"] < 2 or not mask.any():
        return math.inf
    return float(np.nanmax(e[mask]))


def basin_minima(fes: dict, spec) -> dict:
    """Lowest F inside each basin core of a basin set (basins.BasinSet), with its two angles and error:
    {basin: [F, error, angle1, angle2]} or None for a basin with no visited bin."""
    from .basins import core

    edges = fes["edges_deg"]
    centers = (edges[:-1] + edges[1:]) / 2
    a_c, b_c = np.meshgrid(centers, centers, indexing="ij")
    which = core(np.column_stack([a_c.ravel(), b_c.ravel()]), spec)
    f, e = fes["free_energy_kcal"].ravel(), fes["error_kcal"].ravel()
    out = {}
    for i, name in enumerate(spec.names):
        sel = np.nonzero((which == i) & np.isfinite(f))[0]
        if not len(sel):
            out[name] = None
            continue
        k = sel[np.argmin(f[sel])]
        out[name] = [round(float(f[k]), 2), round(float(e[k]), 2) if np.isfinite(e[k]) else None,
                     float(a_c.ravel()[k]), float(b_c.ravel()[k])]
    return out


# ---------------------------------------------------------------- one item, judged

@dataclass
class BackboneSettings:
    basins: str = ""                 # Own regions for rama items; empty: DEFAULT_BASINS_TEXT.
    margin_deg: float = 10.0
    rama: str = "boxes"              # How rama basins are defined: "boxes" (regions, default) or "centres".
    bins_deg: float = 15.0
    min_population: float = 0.05     # Basins at least this populated count as leading for the rate criterion.
    fes_cutoff_kcal: float = 2.0     # The landscape criterion applies where F <= this.
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "BackboneSettings":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def sets_of(state) -> list:
    """The basin sets (basins.BasinSet) of a run, defined as its settings say."""
    from .basins import parse_sets

    bb = BackboneSettings.from_dict(state.backbone_settings)
    return parse_sets(state.backbone, rama=bb.rama, boxes_text=bb.basins, margin_deg=bb.margin_deg)


def diagnose_backbone(items: dict, settings: BackboneSettings, ns_per_replica: float, dt_ns: float,
                      temperature_k: float, min_flips_per_replica: int = 4, precision=None,
                      rate_rel_error: float | None = None, fes_error_kcal: float | None = None):
    """One report per backbone item. `items`: {label: (basins.BasinSet, [angles per replica])}, each angle array
    (frames, 2) in degrees, in the order of the set's dihedrals (phi, psi for rama; psi(i+1), phi(i+2) for turn).

    Returns (reports, details) with details[label] = {"transitions": ..., "landscape": ..., "minima": ...}.
    The verdict is "ok" only when the populations, the leading rates and the landscape all meet their criteria.
    `precision` may be a number (standard error) or an uncertainty.Precision; with a Precision the rate and
    landscape criteria use the same interval width (z), so a run that looks only a few times stays calibrated.
    """
    from .basins import assign, summarize
    from .diagnose import ResidueReport, count_transitions, judge, occupancy, populations_with_errors, rank
    from .uncertainty import Precision

    p = Precision.of(precision)
    z = p.z if p is not None else 1.0
    reports, details = [], {}
    for label, (spec, values) in items.items():
        values = [np.asarray(v, dtype=float).reshape(len(v), -1) for v in values]
        names = spec.names
        states = [assign(v, spec) for v in values]
        occ = [occupancy(s, names) for s in states]
        rep = ResidueReport(chain=spec.chain or "", resseq=spec.resseq, resname="", name=label, quantity="backbone",
                            transitions=[count_transitions(s) for s in states], occupancies=occ,
                            dominant=[max(o, key=o.get) for o in occ], ns_per_replica=ns_per_replica)
        (rep.populations, rep.standard_error, rep.replicas_differ,
         rep.error_from_replicas) = populations_with_errors(states, names)
        judge(rep, min_flips_per_replica, precision)
        est = summarize(states, values, names, dt_ns)      # Free-energy difference, main rates, split-half check.
        rep.error_within, rep.error_between = est["within"], est["between"]
        stats = transition_statistics(states, names, dt_ns)
        rate_err, lead = worst_rate_error(stats, rep.populations, settings.min_population)
        fes = landscape([v[:, 0] for v in values], [v[:, 1] for v in values], temperature_k, settings.bins_deg,
                        n_effective=stats["total_transitions"] + len(states))
        fes_err = worst_landscape_error(fes, settings.fes_cutoff_kcal)
        rep.extra = {"basins": names, "angles": [f"{a} of residue {spec.resseq + off}" for off, a in spec.dihedrals],
                     "dg_kt": est["dg_kt"], "dg_se": est["dg_se"], "rates": est["rates"],
                     "split_js": est.get("split_js"),
                     "rate_rel_error": None if math.isinf(rate_err) else round(rate_err, 3),
                     "leading_transitions": lead,
                     "fes_max_error_kcal": None if math.isinf(fes_err) else round(fes_err, 3),
                     "rates_per_ns": stats["rates_per_ns"], "mean_dwell_ns": stats["mean_dwell_ns"]}
        if rep.verdict == "ok" and rate_rel_error is not None and z * rate_err > rate_rel_error:
            rep.verdict = "imprecise"
            rep.reason += (f" The leading transition rates ({', '.join(lead)}) have a relative error up to "
                           f"{z * rate_err:.2f}" + (f" ({z:.2f} standard errors)" if z != 1 else "")
                           + f", more than the requested {rate_rel_error:g}.")
        elif rep.verdict == "ok" and fes_error_kcal is not None and z * fes_err > fes_error_kcal:
            rep.verdict = "imprecise"
            rep.reason += (f" The free-energy landscape has an error up to {z * fes_err:.2f} kcal/mol where "
                           f"F <= {settings.fes_cutoff_kcal:g}, more than the requested {fes_error_kcal:g}.")
        details[label] = {"transitions": stats, "landscape": fes, "minima": basin_minima(fes, spec),
                          "basins": (basins_text(spec.boxes) if spec.boxes else
                                     "; ".join(f"{k}: centre {c}, core {spec.core_deg:g} deg"
                                               for k, c in spec.centres.items())),
                          "axes": rep.extra["angles"]}
        reports.append(rep)
    return rank(reports), details


def save_details(folder, details: dict, basins_used: str = "") -> list:
    """Write backbone/<label>.json (populations come from status.json), <label>_fes.npz and <label>_fes.csv."""
    import json
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    written = []
    for label, d in details.items():
        stem = label.replace(":", "_")
        (folder / f"{stem}.json").write_text(json.dumps(
            {"residue": label, "basins": d.get("basins", basins_used), "axes": d.get("axes"),
             "transitions": d["transitions"], "basin_minima_kcal": d["minima"],
             "landscape_file": f"{stem}_fes.npz"}, indent=1))
        written += [folder / f"{stem}.json"] + save_landscape(folder, f"{stem}_fes", d["landscape"], label)
    return written


def save_landscape(folder, stem: str, fes: dict, label: str) -> list:
    """<stem>.npz, <stem>.csv (phi, psi, F, error, probability per bin) and <stem>.png if matplotlib is there."""
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{stem}.npz", edges_deg=fes["edges_deg"], free_energy_kcal=fes["free_energy_kcal"],
             error_kcal=fes["error_kcal"], probability=fes["probability"])
    e = fes["edges_deg"]
    c = (e[:-1] + e[1:]) / 2
    rows = ["phi_deg,psi_deg,free_energy_kcal,error_kcal,probability"]
    for i, x in enumerate(c):
        for j, y in enumerate(c):
            f, err, p = fes["free_energy_kcal"][i, j], fes["error_kcal"][i, j], fes["probability"][i, j]
            rows.append(f"{x:g},{y:g},{'' if np.isnan(f) else f'{f:.3f}'},{'' if np.isnan(err) else f'{err:.3f}'},"
                        f"{p:.6g}")
    (folder / f"{stem}.csv").write_text("\n".join(rows) + "\n")
    written = [folder / f"{stem}.npz", folder / f"{stem}.csv"]
    try:   # A picture only if matplotlib happens to be installed.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(fes["free_energy_kcal"].T, origin="lower", extent=[-180, 180, -180, 180], vmax=6)
        ax.set_xlabel("phi (deg)")
        ax.set_ylabel("psi (deg)")
        ax.set_title(f"{label}: F (kcal/mol)")
        fig.colorbar(im)
        fig.savefig(folder / f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(folder / f"{stem}.png")
    except Exception:
        pass
    return written
