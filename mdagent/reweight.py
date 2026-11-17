"""Unbiased chi1 state populations from enhanced sampling, and when to stop it.

Enhanced sampling (enhanced.py) pushes the side chains with a bias, so the
time spent in each state is not the real population. This module removes the
bias with the time-dependent reweighting of Tiwary and Parrinello (J. Phys.
Chem. B 2015, 119, 736): each saved sample gets the weight

    w(t) = exp((V(s(t), t) - c(t)) / kT)

V(s, t) is the bias at the moment the sample was taken, and c(t) is the
reversible work of the bias at that time. The bias is rebuilt exactly by
replaying the Gaussian deposits from the saved chi1 samples (the same
function, enhanced.deposit, that the simulation used). All biased residues
are in one simulation, so a sample's weight uses the sum of all their biases.
The first part of each replica is left out, because the bias grows fastest
there.

Each state gets a population (g+, t, g-; the same state cores as the plain MD
diagnosis) and a free energy relative to the most populated state.

When to stop: more sampling no longer changes the result when, for every
residue, the replicas agree on every population within `tol`, and the
combined estimate changed by less than tol / 2 since the previous check.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .diagnose import STATE_NAMES, assign_states
from .enhanced import KB_KJ, MetadSettings, deposit

KJ_PER_KCAL = 4.184


@dataclass
class StopRule:
    tol: float = 0.10            # Largest allowed difference in a population (0 to 1).
    min_ns: float = 5.0          # Never stop before this length per replica.
    skip_fraction: float = 0.1   # Leave out this first part of each replica.


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def replay(chi: np.ndarray, s: MetadSettings, kt: float) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the bias along a replica. Returns, for each sample, the total bias acting
    when it was taken (kJ/mol) and c(t) (kJ/mol)."""
    n, m = chi.shape
    grid = np.linspace(-math.pi, math.pi, s.grid_points)
    bias = np.zeros((m, s.grid_points))
    a = 1.0 / (kt * (s.bias_factor - 1.0))
    v, c = np.empty(n), np.empty(n)
    for t in range(n):
        x = chi[t]
        v[t] = sum(float(np.interp(x[i], grid, bias[i])) for i in range(m))
        g = bias[:, :-1]  # The last grid point repeats the first.
        c[t] = kt * float(np.sum(_logsumexp(s.bias_factor * a * g) - _logsumexp(a * g)))
        deposit(bias, grid, x, s, kt)  # This sample's Gaussian acts only on later samples.
    return v, c


def weighted_populations(chi_rad: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    states = assign_states(np.degrees(chi_rad))
    ok = states >= 0
    total = float(np.sum(weights[ok]))
    if total <= 0:
        return {name: float("nan") for name in STATE_NAMES}
    return {name: float(np.sum(weights[ok & (states == k)]) / total) for k, name in enumerate(STATE_NAMES)}


def free_energies_kcal(pops: dict[str, float], kt: float) -> dict[str, float | None]:
    """Free energy of each state relative to the most populated one (kcal/mol). None if never seen."""
    top = max(pops.values())
    return {k: (round(-kt * math.log(p / top) / KJ_PER_KCAL, 2) + 0.0 if p > 0 else None) for k, p in pops.items()}


def analyze(run_dir: Path, labels: list[str], n_replicas: int, s: MetadSettings, temperature_k: float,
            rule: StopRule = StopRule()) -> dict | None:
    """Reweighted populations of each biased residue, per replica and combined. None if no data yet."""
    kt = KB_KJ * temperature_k
    per_rep, n_eff, ns = [], [], []
    for r in range(n_replicas):
        path = Path(run_dir) / "enhanced" / f"replica_{r}_chi.npy"
        if not path.exists():
            continue
        chi = np.load(path)
        if chi.ndim != 2 or len(chi) < 10:
            continue
        v, c = replay(chi, s, kt)
        start = int(len(chi) * rule.skip_fraction)
        logw = (v[start:] - c[start:]) / kt
        w = np.exp(logw - logw.max())
        n_eff.append(round(float(w.sum() ** 2 / np.sum(w * w)), 1))  # Effective number of samples.
        ns.append(len(chi) * s.pace_ps / 1000.0)
        per_rep.append([weighted_populations(chi[start:, i], w) for i in range(len(labels))])
    if not per_rep:
        return None
    residues = []
    for i, label in enumerate(labels):
        reps = [p[i] for p in per_rep]
        pooled = {k: float(np.nanmean([p[k] for p in reps])) for k in STATE_NAMES}
        spread = max(max(p[k] for p in reps) - min(p[k] for p in reps) for k in STATE_NAMES) if len(reps) > 1 \
            else float("nan")
        residues.append({"residue": label, "populations": {k: round(v, 3) for k, v in pooled.items()},
                         "free_energy_kcal": free_energies_kcal(pooled, kt),
                         "per_replica": [{k: round(v, 3) for k, v in p.items()} for p in reps],
                         "replica_spread": round(spread, 3)})
    return {"ns_per_replica": round(min(ns), 6), "n_replicas": len(per_rep), "effective_samples": n_eff,
            "method": "Tiwary-Parrinello time-dependent reweighting; first "
                      f"{rule.skip_fraction:.0%} of each replica left out", "residues": residues}


def converged(now: dict, before: dict | None, rule: StopRule) -> tuple[bool, list[str]]:
    """True when more sampling no longer changes the result. Also returns the residues that still change."""
    open_ = []
    prev = {r["residue"]: r["populations"] for r in (before or {}).get("residues", [])}
    for r in now["residues"]:
        spread = r["replica_spread"]
        disagree = not math.isnan(spread) and spread > rule.tol
        old = prev.get(r["residue"])
        moving = old is None or max(abs(r["populations"][k] - old[k]) for k in STATE_NAMES) > rule.tol / 2
        if disagree or moving:
            open_.append(r["residue"])
    return (not open_ and now["ns_per_replica"] >= rule.min_ns - 1e-9), open_


def format_table(result: dict) -> str:
    lines = [f"Reweighted chi1 populations after {result['ns_per_replica']:g} ns per replica "
             f"({result['n_replicas']} replicas; {result['method']}).",
             f"{'residue':<10}{'g+':>7}{'t':>7}{'g-':>7}   {'free energy (kcal/mol) g+ / t / g-':<36}replica spread"]
    for r in result["residues"]:
        p, g = r["populations"], r["free_energy_kcal"]
        fe = " / ".join("-" if g[k] is None else f"{g[k]:.2f}" for k in STATE_NAMES)
        spread = "-" if math.isnan(r["replica_spread"]) else f"{r['replica_spread']:.2f}"
        lines.append(f"{r['residue']:<10}{p['g+']:>7.2f}{p['t']:>7.2f}{p['g-']:>7.2f}   {fe:<36}{spread}")
    return "\n".join(lines)


def save(run_dir: Path, result: dict) -> Path:
    path = Path(run_dir) / "enhanced" / "populations.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(path)
    return path
