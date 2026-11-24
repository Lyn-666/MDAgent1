"""Structure stability: does a protein (or a region of it) keep its structure at this temperature?

Two continuous observables are followed every round:

- RMSD: the root-mean-square deviation from the starting structure (protein.pdb), after superposition on the same
  atoms. The atoms are the C-alpha atoms of the region, or all heavy atoms of the region when it has fewer than
  `min_ca` residues (three C-alpha atoms say little about how a core is packed).
- Rg: the radius of gyration of the same atoms.

For each, the run reports the mean after the non-equilibrated start (automated equilibration detection, one start
for all replicas), with an error that allows for time correlation (block averaging inside replicas, the spread
between replicas, Student's t when that spread is real). The item is precise enough when the interval half-width
is at most `tolerance_nm` (default 0.02 nm).

Steady state is an equivalence check, not "no significant difference": the kept frames of every replica are cut in
two halves; the mean of the second halves minus the mean of the first halves, plus the interval half-width of that
difference, must be at most `drift_factor` x `tolerance_nm` (default 3 x 0.02 = 0.06 nm). A short run cannot pass it
by being imprecise.

Whether the structure stays native-like: the time fraction with RMSD below `rmsd_cutoff_nm` (default 0.3 nm, with a
0.05 nm buffer so that crossing back and forth at the threshold is not counted as a change). If no replica ever
leaves, MD cannot prove it never will; the report gives the 95 % upper bound of the departure rate, 3 / (time
watched), so the answer reads "no departure in T ns; the rate is at most 3/T".

The free-energy surface F(RMSD, Rg) = -kT ln p of the kept frames, with an error from the spread between replicas.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

KB_KCAL = 0.0019872041   # kcal/mol/K


@dataclass
class StabilitySettings:
    tolerance_nm: float = 0.02       # interval half-width asked for the mean RMSD and the mean Rg
    rmsd_cutoff_nm: float = 0.3      # above this RMSD the structure counts as departed from the start
    buffer_nm: float = 0.05          # hysteresis around the cutoff
    drift_factor: float = 3.0        # steady state: drift plus its half-width at most this x tolerance_nm
    cluster_cutoff_nm: float = 0.15  # GROMOS clustering cutoff (clustering.py)
    min_ca: int = 5                  # fewer residues than this: use all heavy atoms of the region
    max_departure_rate_per_us: float = 10.0   # stop only when the 95 % upper bound of the departure rate is this low

    @classmethod
    def from_dict(cls, d: dict | None) -> "StabilitySettings":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


STATES = ["native", "departed"]
ITEMS = ("RMSD", "Rg")


def atom_selection(top, keys: list, min_ca: int = 5) -> tuple[list[int], str]:
    """Atom indices (in `top`) for the stability observables, and a description. keys: [(chain, resseq)] or empty
    for the whole protein."""
    wanted = {(c, n) for c, n in keys}
    protein = [r for r in top.residues if r.is_protein]
    residues = [r for r in protein if not wanted or (r.chain.chain_id, r.resSeq) in wanted]
    if not residues:
        raise ValueError("no protein residues in the region for the stability observables.")
    whole = len(residues) == len(protein)
    region = ("the whole protein" if whole else
              ", ".join(f"{r.chain.chain_id}:{r.name}{r.resSeq}" for r in residues[:12])
              + (" ..." if len(residues) > 12 else ""))
    if len(residues) >= min_ca:
        atoms = [a.index for r in residues for a in r.atoms if a.name == "CA"]
        return atoms, f"C-alpha atoms of {region} ({len(atoms)} atoms)"
    atoms = [a.index for r in residues for a in r.atoms if a.element is not None and a.element.symbol != "H"]
    return atoms, f"heavy atoms of {region} ({len(atoms)} atoms; fewer than {min_ca} residues)"


def series(trajs: list, reference) -> list[tuple[np.ndarray, np.ndarray]]:
    """(RMSD, Rg) in nm, frame by frame, per replica. `trajs` and `reference` hold the same atoms."""
    import mdtraj as md

    out = []
    for t in trajs:
        out.append((md.rmsd(t, reference, 0).astype(float), md.compute_rg(t).astype(float)))
    return out


def _mean_errors(values: list[np.ndarray]) -> dict:
    """Mean over replicas with its error parts (uncertainty.py), per-replica means and independent samples."""
    from .uncertainty import block_standard_error, error_parts, statistical_inefficiency

    means = [float(np.mean(v)) if len(v) else float("nan") for v in values]
    errs = [block_standard_error(v) for v in values]
    within, between = error_parts(means, errs)
    neff = [int(len(v) / statistical_inefficiency(v)) if len(v) > 1 else 0 for v in values]
    return {"mean": float(np.nanmean(means)), "se": max(within, between) if len(values) > 1 else
            (errs[0] if errs and math.isfinite(errs[0]) else float("nan")),
            "within": within, "between": between, "per_replica": means, "independent_samples": neff}


def drift(values: list[np.ndarray], z: float, tolerance_nm: float, factor: float) -> dict:
    """Second-half mean minus first-half mean of the kept frames, its interval half-width, and whether the
    difference is within factor x tolerance with confidence (an equivalence check).

    The halves are paired within each replica, so a replica that sits at a different level than the others does not
    look like drift: d_i = mean(second half_i) - mean(first half_i), averaged over replicas. Its error is the larger
    of the within-replica error (block errors of the two halves) and the spread of d_i between replicas."""
    from .uncertainty import block_standard_error, error_parts

    d_i, e_i = [], []
    for v in values:
        if len(v) < 4:
            continue
        a, b = v[: len(v) // 2], v[len(v) // 2:]
        d_i.append(float(np.mean(b) - np.mean(a)))
        e_i.append(math.hypot(block_standard_error(a), block_standard_error(b)))
    if not d_i:
        return {"drift_nm": float("nan"), "halfwidth_nm": float("nan"), "steady": False}
    d = float(np.mean(d_i))
    if len(d_i) > 1:
        within, between = error_parts(d_i, e_i)
        se = max(within, between)
    else:
        se = e_i[0]
    half = z * se if math.isfinite(se) else float("inf")
    return {"drift_nm": d, "halfwidth_nm": half, "steady": bool(abs(d) + half <= factor * tolerance_nm),
            "limit_nm": factor * tolerance_nm, "per_replica_nm": d_i}


def rate_upper(k: int, confidence: float = 0.95) -> float:
    """Upper confidence bound of a Poisson mean after seeing k events: 3.0 for k = 0, 4.74 for k = 1, ..."""
    from scipy.stats import chi2

    return 0.5 * float(chi2.ppf(confidence, 2 * k + 2))


def native(rmsd: list[np.ndarray], s: StabilitySettings, dt_ns: float) -> dict:
    """Time fraction native-like (RMSD below the cutoff, with hysteresis), the departures, and the 95 % upper bound
    of the departure rate (Poisson). needed_total_ns: the watched time (all replicas) at which that bound reaches
    s.max_departure_rate_per_us, if no further departure happens."""
    from .observables import two_state

    fractions, departures, watched = [], [], 0.0
    for v in rmsd:
        st = two_state(v, s.rmsd_cutoff_nm, s.buffer_nm)        # 0: near (native-like), 1: far
        st = st[st >= 0]
        fractions.append(float(np.mean(st == 0)) if len(st) else (1.0 if np.all(v <= s.rmsd_cutoff_nm) else 0.0))
        departures.append(int(np.count_nonzero((st[:-1] == 0) & (st[1:] == 1))) if len(st) > 1 else 0)
        watched += len(v) * dt_ns
    out = {"fraction_per_replica": fractions, "fraction": float(np.nanmean(fractions)) if fractions else float("nan"),
           "departures_per_replica": departures, "watched_ns": watched}
    k = sum(departures)
    if watched > 0:
        out["departure_rate_per_us"] = k / watched * 1000.0
        out["departure_rate_upper_per_us"] = rate_upper(k) / watched * 1000.0
    out["max_departure_rate_per_us"] = s.max_departure_rate_per_us
    out["needed_total_ns"] = rate_upper(k) / s.max_departure_rate_per_us * 1000.0
    return out


def landscape(rmsd: list[np.ndarray], rg: list[np.ndarray], temperature_k: float, bins: int = 25) -> dict:
    """F(RMSD, Rg) in kcal/mol from the kept frames of all replicas; errors from the spread between replicas."""
    allr, allg = np.concatenate(rmsd), np.concatenate(rg)
    if allr.size < 2:
        return {}
    pad_r, pad_g = 0.02 * (np.ptp(allr) or 0.1), 0.02 * (np.ptp(allg) or 0.1)
    er = np.linspace(max(0.0, allr.min() - pad_r), allr.max() + pad_r, bins + 1)
    eg = np.linspace(allg.min() - pad_g, allg.max() + pad_g, bins + 1)
    hists = []
    for r, g in zip(rmsd, rg):
        h, _, _ = np.histogram2d(r, g, bins=[er, eg])
        hists.append(h / h.sum() if h.sum() else h)
    hists = np.array(hists)
    p = hists.mean(axis=0)
    kt = KB_KCAL * temperature_k
    f = np.full(p.shape, np.nan)
    seen = p > 0
    f[seen] = -kt * np.log(p[seen] / p[seen].max())
    err = np.full(p.shape, np.nan)
    if len(hists) > 1:
        se = hists.std(axis=0, ddof=1) / math.sqrt(len(hists))
        err[seen] = kt * se[seen] / p[seen]
    return {"rmsd_edges_nm": er, "rg_edges_nm": eg, "free_energy_kcal": f, "error_kcal": err, "probability": p}


def analyze(rmsd: list[np.ndarray], rg: list[np.ndarray], s: StabilitySettings, precision, dt_ns: float,
            ns_per_replica: float, min_samples: int = 4, burn_in: str = "auto", temperature_k: float = 300.0,
            selection: str = "") -> tuple[list, dict]:
    """Two reports (RMSD, Rg) in the form the agent judges, and the details for the report and the figures."""
    from .diagnose import ResidueReport
    from .uncertainty import Precision, joint_equilibration_start_values, replica_spread_real

    p = Precision.of(precision) or Precision(s.tolerance_nm)
    p = Precision(epsilon=s.tolerance_nm, criterion=p.criterion, alpha=p.alpha, looks=p.looks)
    t0 = (max(joint_equilibration_start_values(rmsd), joint_equilibration_start_values(rg))
          if burn_in == "auto" else 0)
    kept_r, kept_g = [v[t0:] for v in rmsd], [v[t0:] for v in rg]
    nat = native(kept_r, s, dt_ns)
    reports, details = [], {"selection": selection, "burn_in_frames": t0, "burn_in_ns": t0 * dt_ns,
                            "settings": asdict(s), "native": nat, "items": {}}
    for name, kept in (("RMSD", kept_r), ("Rg", kept_g)):
        m = _mean_errors(kept)
        n = len(kept)
        real = n > 1 and replica_spread_real(m["within"], m["between"], n)
        half = p.halfwidth(m["se"], df=n - 1 if real else None) if math.isfinite(m["se"]) else float("inf")
        dr = drift(kept, p.factor(n - 1 if real else None), s.tolerance_nm, s.drift_factor)
        enough = bool(m["independent_samples"]) and min(m["independent_samples"]) >= min_samples
        clear_drift = math.isfinite(dr["drift_nm"]) and abs(dr["drift_nm"]) > dr.get("limit_nm", float("inf"))
        if clear_drift:
            verdict, why = "drifting", (f"still drifting: the second half differs from the first by "
                                        f"{dr['drift_nm']:+.3f} nm, more than the allowed {dr['limit_nm']:.3f} nm")
        elif not enough:
            verdict, why = "rare_flips", (f"too few independent samples per replica ({m['independent_samples']}) to "
                                          "trust the error yet")
        elif not dr["steady"]:
            verdict, why = "drifting", (f"not steady yet: the second half differs from the first by "
                                        f"{dr['drift_nm']:+.3f} ± {dr['halfwidth_nm']:.3f} nm (allowed "
                                        f"{dr['limit_nm']:.3f} nm)")
        elif half > s.tolerance_nm:
            verdict, why = "imprecise", f"steady, but the interval half-width is {half:.3f} nm (asked {s.tolerance_nm:g} nm)"
        elif name == "RMSD" and nat.get("departure_rate_upper_per_us", float("inf")) > s.max_departure_rate_per_us:
            # Precise means say nothing about longer times. "Stable" needs the structure watched long enough that
            # leaving it would have been seen: the 95 % bound of the departure rate must reach the goal.
            verdict, why = "too_short", (
                f"steady and precise, but watched only {nat['watched_ns']:.3g} ns in all: the departure rate is at "
                f"most {nat.get('departure_rate_upper_per_us', float('inf')):.3g} per µs (95 %), the goal is at most "
                f"{s.max_departure_rate_per_us:g} per µs, which needs about {nat['needed_total_ns']:.3g} ns in all")
        else:
            verdict, why = "ok", "steady and precise enough"
        reason = f"{name} {m['mean']:.3f} ± {half:.3f} nm (interval half-width); {why}."
        rep = ResidueReport(chain="", resseq=0, resname="", name=name, quantity="stability",
                            transitions=list(m["independent_samples"]),
                            occupancies=[{"mean": x} for x in m["per_replica"]],
                            dominant=[f"{x:.3f}" for x in m["per_replica"]], ns_per_replica=ns_per_replica,
                            verdict=verdict, reason=reason, populations={"mean": m["mean"]},
                            standard_error={"mean": m["se"]}, halfwidth={"mean": half},
                            replicas_differ=bool(real), error_from_replicas=m["between"] > m["within"],
                            error_within=m["within"], error_between=m["between"], burn_in_ns=t0 * dt_ns,
                            extra={"unit": "nm", "drift": dr, "selection": selection,
                                   **({"rate_needs_total_ns": nat["needed_total_ns"]} if name == "RMSD" else {})})
        reports.append(rep)
        details["items"][name] = {**m, "halfwidth_nm": half, "drift": dr, "verdict": verdict}
    details["landscape"] = landscape(kept_r, kept_g, temperature_k)
    return reports, details


# ------------------------------------------------------------------------------------------------------------------
# Output: stability/summary.json, series and landscape as CSV, and two SVG figures.

def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def save(folder: Path, rmsd: list[np.ndarray], rg: list[np.ndarray], details: dict, dt_ns: float,
         max_points: int = 1500) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    summary = {k: v for k, v in details.items() if k != "landscape"}
    (folder / "summary.json").write_text(json.dumps(_plain(summary), indent=1))
    paths.append(folder / "summary.json")
    rows = ["replica,time_ns,rmsd_nm,rg_nm"]
    for i, (r, g) in enumerate(zip(rmsd, rg)):
        step = max(1, len(r) // max_points)
        rows += [f"{i},{k * dt_ns:.4f},{r[k]:.4f},{g[k]:.4f}" for k in range(0, len(r), step)]
    (folder / "rmsd_rg.csv").write_text("\n".join(rows) + "\n")
    (folder / "rmsd_rg.svg").write_text(series_svg(rmsd, rg, dt_ns, details))
    paths += [folder / "rmsd_rg.csv", folder / "rmsd_rg.svg"]
    fes = details.get("landscape") or {}
    if fes:
        er, eg, f, e = fes["rmsd_edges_nm"], fes["rg_edges_nm"], fes["free_energy_kcal"], fes["error_kcal"]
        rows = ["rmsd_nm,rg_nm,free_energy_kcal,error_kcal"]
        for i in range(len(er) - 1):
            for j in range(len(eg) - 1):
                if np.isfinite(f[i, j]):
                    rows.append(f"{(er[i] + er[i + 1]) / 2:.4f},{(eg[j] + eg[j + 1]) / 2:.4f},{f[i, j]:.3f},"
                                + (f"{e[i, j]:.3f}" if np.isfinite(e[i, j]) else ""))
        (folder / "fes_rmsd_rg.csv").write_text("\n".join(rows) + "\n")
        (folder / "fes_rmsd_rg.svg").write_text(fes_svg(fes))
        paths += [folder / "fes_rmsd_rg.csv", folder / "fes_rmsd_rg.svg"]
    return paths


_STYLE = ('<style>svg{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--grid:#e4e3de;--s1:#2a78d6;--s2:#eb6834;'
          '--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300}'
          '@media (prefers-color-scheme: dark){svg{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--grid:#33332f}}'
          'text{fill:var(--muted)} .h{fill:var(--fg);font-weight:600}</style>')


def series_svg(rmsd: list[np.ndarray], rg: list[np.ndarray], dt_ns: float, details: dict) -> str:
    """Two panels: RMSD and Rg against time, one line per replica; the burn-in is shaded."""
    W, H, L, R, T, B, gap = 760, 460, 64, 20, 36, 44, 46
    ph = (H - T - B - gap) / 2
    t_max = max((len(v) for v in rmsd), default=1) * dt_ns or 1.0
    burn = details.get("burn_in_ns", 0.0)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         'font-family="system-ui, sans-serif" font-size="12"><title>RMSD and Rg against time</title>', _STYLE,
         f'<rect width="{W}" height="{H}" fill="var(--bg)"/>']
    for k, (name, data) in enumerate((("RMSD (nm)", rmsd), ("Rg (nm)", rg))):
        top = T + k * (ph + gap)
        vals = np.concatenate([v for v in data if len(v)]) if any(len(v) for v in data) else np.array([0.0, 1.0])
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-6:
            lo, hi = lo - 0.05, hi + 0.05
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def X(t):
            return L + (W - L - R) * t / t_max

        def Y(v, lo=lo, hi=hi, top=top):
            return top + ph * (1 - (v - lo) / (hi - lo))
        p.append(f'<text class="h" x="{L}" y="{top - 8:.0f}">{name}</text>')
        if burn > 0:
            p.append(f'<rect x="{L}" y="{top:.0f}" width="{X(burn) - L:.1f}" height="{ph:.0f}" fill="var(--grid)" '
                     'opacity="0.6"><title>left out: not yet equilibrated</title></rect>')
        for q in range(5):
            v = lo + (hi - lo) * q / 4
            p.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/>'
                     f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
        for i, v in enumerate(data):
            if not len(v):
                continue
            step = max(1, len(v) // 800)
            pts = " ".join(f"{X(j * dt_ns):.1f},{Y(v[j]):.1f}" for j in range(0, len(v), step))
            p.append(f'<polyline points="{pts}" fill="none" stroke="var(--s{i % 6 + 1})" stroke-width="1.2">'
                     f'<title>replica {i}</title></polyline>')
    for q in range(6):
        t = t_max * q / 5
        p.append(f'<text x="{L + (W - L - R) * q / 5:.1f}" y="{H - B + 18}" text-anchor="middle">{t:.3g}</text>')
    p.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 10}" text-anchor="middle">time per replica (ns); '
             'shaded: left out as not yet equilibrated</text></svg>')
    return "\n".join(p)


def fes_svg(fes: dict) -> str:
    """Heat map of F(RMSD, Rg) in kcal/mol (darker is lower); empty bins are blank."""
    er, eg, f = fes["rmsd_edges_nm"], fes["rg_edges_nm"], fes["free_energy_kcal"]
    W, H, L, R, T, B = 560, 500, 70, 90, 36, 50
    nx, ny = len(er) - 1, len(eg) - 1
    cw, ch = (W - L - R) / nx, (H - T - B) / ny
    fmax = float(np.nanmax(f)) if np.isfinite(f).any() else 1.0
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         'font-family="system-ui, sans-serif" font-size="12"><title>Free-energy surface F(RMSD, Rg)</title>', _STYLE,
         f'<rect width="{W}" height="{H}" fill="var(--bg)"/>',
         f'<text class="h" x="{L}" y="22">F(RMSD, Rg), kcal/mol</text>']
    for i in range(nx):
        for j in range(ny):
            v = f[i, j]
            if not np.isfinite(v):
                continue
            shade = 1 - min(v / max(fmax, 1e-9), 1.0)
            p.append(f'<rect x="{L + i * cw:.1f}" y="{T + (ny - 1 - j) * ch:.1f}" width="{cw + 0.5:.1f}" '
                     f'height="{ch + 0.5:.1f}" fill="var(--s1)" fill-opacity="{0.12 + 0.88 * shade:.2f}">'
                     f'<title>RMSD {(er[i] + er[i + 1]) / 2:.3f} nm, Rg {(eg[j] + eg[j + 1]) / 2:.3f} nm: '
                     f'F {v:.2f} kcal/mol</title></rect>')
    for q in range(5):
        x = er[0] + (er[-1] - er[0]) * q / 4
        y = eg[0] + (eg[-1] - eg[0]) * q / 4
        p.append(f'<text x="{L + (W - L - R) * q / 4:.1f}" y="{H - B + 18}" text-anchor="middle">{x:.2f}</text>')
        p.append(f'<text x="{L - 6}" y="{H - B - (H - T - B) * q / 4 + 4:.1f}" text-anchor="end">{y:.2f}</text>')
    p.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 12}" text-anchor="middle">RMSD from the start (nm)</text>')
    p.append(f'<text x="16" y="{(T + H - B) / 2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 16 {(T + H - B) / 2:.0f})">Rg (nm)</text>')
    p.append(f'<text x="{W - R + 10}" y="{T + 12}">darker:</text><text x="{W - R + 10}" y="{T + 28}">lower F</text>')
    p.append(f'<text x="{W - R + 10}" y="{T + 52}">0 to {fmax:.1f}</text><text x="{W - R + 10}" y="{T + 68}">kcal/mol</text>')
    p.append("</svg>")
    return "\n".join(p)


def format_report(reports: list, details: dict | None = None) -> str:
    """The round table for the log."""
    lines = ["Structure stability" + (f" ({details['selection']})" if details and details.get("selection") else ""),
             f"{'item':<6}{'verdict':<12}{'mean (nm)':<12}{'half-width':<12}{'drift (nm)':<16}"
             "independent samples per replica"]
    for r in reports:
        dr = (r.extra or {}).get("drift", {})
        drift_text = (f"{dr.get('drift_nm', float('nan')):+.3f}" if dr else "-")
        half = (r.halfwidth or {}).get("mean", float("nan"))
        lines.append(f"{r.name:<6}{r.verdict:<12}{r.populations['mean']:<12.3f}{half:<12.3f}{drift_text:<16}"
                     f"{','.join(map(str, r.transitions))}")
    if details and details.get("native"):
        nat = details["native"]
        line = (f"Native-like (RMSD below {details['settings']['rmsd_cutoff_nm']:g} nm): "
                f"{nat['fraction']:.0%} of the kept time; departures per replica {nat['departures_per_replica']}.")
        if "departure_rate_upper_per_us" in nat:
            line += (f" {sum(nat['departures_per_replica'])} departures in {nat['watched_ns']:.3g} ns: the rate is at "
                     f"most {nat['departure_rate_upper_per_us']:.3g} per µs (95 %; goal at most "
                     f"{nat.get('max_departure_rate_per_us', 10):g}).")
        lines.append(line)
    return "\n".join(lines)
