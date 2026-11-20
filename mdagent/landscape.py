"""The phi/psi free-energy landscape of each backbone observable, as a table and a figure.

F = -ln(P / P_max) in kT, from the 10 x 10 degree histogram of all replicas and rounds. Empty bins
have no free energy (they were never visited, which says nothing about how high they are). The
figure shades bins in 1 kT bands, darkest at the minimum; the CSV has every bin.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

BINS = 36
BAND_KT = 1.0
TOP_KT = 6.0
# Single-hue blue ramp (dataviz reference palette), steps 700 -> 100: the minimum is darkest on light.
RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]


def free_energy(hist: np.ndarray) -> np.ndarray:
    """kT above the most populated bin; NaN where nothing was seen."""
    h = np.asarray(hist, float)
    out = np.full(h.shape, np.nan)
    seen = h > 0
    if seen.any():
        out[seen] = -np.log(h[seen] / h[seen].max())
    return out


def compute(run_dir) -> dict:
    """label -> {"axes": [...], "hist": 36x36 array, "frames": n} for every backbone observable of a run."""
    import mdtraj as md

    from .agent import RunState
    from .basins import angles, dihedral_atoms, parse_sets
    from .diagnose import load_run

    run_dir = Path(run_dir)
    state = RunState.load(run_dir)
    sets = parse_sets(state.backbone)
    if not sets or not state.rounds_ns:
        return {}
    top = md.load_topology(str(run_dir / "protein.pdb"))
    quads = [dihedral_atoms(top, s) for s in sets]
    atoms = sorted({a for q in quads for quad in q for a in quad})
    where = {a: k for k, a in enumerate(atoms)}
    trajs = load_run(run_dir, state.target, len(state.rounds_ns), atoms=atoms)
    out = {}
    for s, q in zip(sets, quads):
        local = [[where[a] for a in quad] for quad in q]
        vals = np.concatenate([angles(t, local) for t in trajs])
        h, _, _ = np.histogram2d(vals[:, 0], vals[:, 1], bins=BINS, range=[[-180, 180], [-180, 180]])
        out[s.label] = {"axes": [f"{a} {s.resseq + off}" for off, a in s.dihedrals], "hist": h,
                        "frames": int(len(vals)), "centres": s.centres, "core_deg": s.core_deg}
    return out


def svg(label: str, axes: list, hist: np.ndarray, centres: dict | None = None, core_deg: float = 50.0) -> str:
    F = free_energy(hist)
    S, L, T, R, B = 360, 56, 34, 120, 46
    W, H = L + S + R, T + S + B
    cell = S / BINS
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-labelledby="t" font-family="system-ui, sans-serif" font-size="11">',
         f'<title id="t">Free-energy landscape of {label}</title>',
         "<style>svg{--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e4e3de;"
         + "".join(f"--b{i}:{c};" for i, c in enumerate(RAMP)) + "}"
         "@media (prefers-color-scheme: dark){svg{--surface-1:#1a1a19;--text-primary:#f0efec;"
         "--text-secondary:#c3c2b7;--grid:#33332f;"
         + "".join(f"--b{i}:{c};" for i, c in enumerate(reversed(RAMP))) + "}}"
         "text{fill:var(--text-secondary)}.h{fill:var(--text-primary);font-weight:600}</style>",
         f'<rect width="{W}" height="{H}" fill="var(--surface-1)"/>',
         f'<text class="h" x="{L}" y="20">{label}: free energy (kT) of the {axes[0]} / {axes[1]} angles</text>',
         f'<rect x="{L}" y="{T}" width="{S}" height="{S}" fill="none" stroke="var(--grid)"/>']
    for i in range(BINS):          # x = first angle
        for j in range(BINS):      # y = second angle
            f = F[i, j]
            if not np.isfinite(f):
                continue
            band = min(len(RAMP) - 1, int(f // BAND_KT))
            x, y = L + i * cell, T + S - (j + 1) * cell
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell + 0.2:.1f}" height="{cell + 0.2:.1f}" '
                     f'fill="var(--b{band})"><title>{axes[0]} {-180 + 10 * i}..{-170 + 10 * i}°, {axes[1]} '
                     f'{-180 + 10 * j}..{-170 + 10 * j}°: {f:.1f} kT</title></rect>')
    for v in (-180, -90, 0, 90, 180):
        x, y = L + (v + 180) / 360 * S, T + S - (v + 180) / 360 * S
        p.append(f'<text x="{x:.1f}" y="{T + S + 16}" text-anchor="middle">{v}</text>')
        p.append(f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{v}</text>')
    p.append(f'<text x="{L + S / 2:.0f}" y="{H - 8}" text-anchor="middle">{axes[0]} (degrees)</text>')
    p.append(f'<text x="14" y="{T + S / 2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 14 {T + S / 2:.0f})">{axes[1]} (degrees)</text>')
    for name, c in (centres or {}).items():   # basin cores, so the populations can be read against the map
        x, y = L + (c[0] + 180) / 360 * S, T + S - (c[1] + 180) / 360 * S
        r = core_deg / 360 * S
        ty = y - r - 4 if y - r - 4 > T + 12 else y - r + 14   # inside the circle when it touches the top
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="none" stroke="var(--text-primary)" '
                 f'stroke-dasharray="3 3"/><text class="h" x="{x:.1f}" y="{ty:.1f}" '
                 f'text-anchor="middle">{name}</text>')
    ly = T + 4
    p.append(f'<text class="h" x="{L + S + 16}" y="{ly}">kT</text>')
    for i in range(len(RAMP)):
        ly += 20
        top = "" if i == len(RAMP) - 1 else f"–{(i + 1) * BAND_KT:g}"
        p.append(f'<rect x="{L + S + 16}" y="{ly - 10}" width="14" height="14" rx="2" fill="var(--b{i})"/>'
                 f'<text x="{L + S + 36}" y="{ly + 1}">{i * BAND_KT:g}{top}{"+" if not top else ""}</text>')
    p.append(f'<text x="{L + S + 16}" y="{ly + 26}">blank: not seen</text>')
    p.append("</svg>")
    return "\n".join(p)


def write(run_dir) -> list[Path]:
    """landscape_<label>.csv and .svg for every backbone observable. Returns the paths written."""
    run_dir = Path(run_dir)
    paths = []
    for label, d in compute(run_dir).items():
        stem = "landscape_" + label.replace(":", "_")
        F = free_energy(d["hist"])
        with open(run_dir / f"{stem}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([f"{d['axes'][0]} from", f"{d['axes'][1]} from", "frames", "free_energy_kT"])
            for i in range(BINS):
                for j in range(BINS):
                    w.writerow([-180 + 10 * i, -180 + 10 * j, int(d["hist"][i, j]),
                                "" if not math.isfinite(F[i, j]) else round(float(F[i, j]), 3)])
        (run_dir / f"{stem}.svg").write_text(svg(label, d["axes"], d["hist"], d["centres"], d["core_deg"]))
        paths += [run_dir / f"{stem}.csv", run_dir / f"{stem}.svg"]
    return paths
