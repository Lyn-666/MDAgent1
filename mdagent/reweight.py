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

Errors come from block averaging inside each replica and from the spread
between replicas (uncertainty.py). Each residue then gets a verdict (assess):
the goal is reached only when the error is at most the user's precision, the
replicas agree, every replica crossed between states often enough, and the
weights are not degenerate. Populations that stop changing are only a diagnosis:
all replicas can be trapped in the same state and still agree.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .diagnose import STATE_NAMES, assign_states, count_transitions
from .enhanced import KB_KJ, MetadSettings, deposit
from .uncertainty import block_standard_error, combine_replicas, effective_samples, error_parts

KJ_PER_KCAL = 4.184


@dataclass
class StopRule:
    min_ns: float = 5.0             # Never judge a residue "reached" before this length per replica.
    skip_fraction: float = 0.1      # Leave out this first part of each replica.
    min_transitions: int = 10       # Each replica must cross between states at least this often.
    min_effective_samples: float = 50.0  # Fewer effective samples: the weights are degenerate.


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


def free_energies_kcal(pops: dict[str, float], kt: float) -> dict[str, float | None]:
    """Free energy of each state relative to the most populated one (kcal/mol). None if never seen."""
    top = max(pops.values())
    return {k: (round(-kt * math.log(p / top) / KJ_PER_KCAL, 2) + 0.0 if p > 0 else None) for k, p in pops.items()}


def _replica(chi_rad: np.ndarray, weights: np.ndarray) -> dict:
    """Populations, their errors and the number of transitions for one residue in one replica."""
    states = assign_states(np.degrees(chi_rad))
    ok = states >= 0
    w = np.where(ok, weights, 0.0)
    total = float(w.sum())
    pops, errs = {}, {}
    for k, name in enumerate(STATE_NAMES):
        hit = (states == k).astype(float)
        pops[name] = float(np.sum(w * hit) / total) if total > 0 else float("nan")
        errs[name] = block_standard_error(hit[ok], weights[ok])
    return {"populations": pops, "errors": errs, "transitions": count_transitions(states)}


def analyze(run_dir: Path, labels: list[str], n_replicas: int, s: MetadSettings, temperature_k: float,
            rule: StopRule = StopRule()) -> dict | None:
    """Reweighted populations of each biased residue, with errors, per replica and combined.

    None if there is no data yet.
    """
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
        n_eff.append(round(effective_samples(w), 1))
        ns.append(len(chi) * s.pace_ps / 1000.0)
        per_rep.append([_replica(chi[start:, i], w) for i in range(len(labels))])
    if not per_rep:
        return None
    residues = []
    for i, label in enumerate(labels):
        reps = [p[i] for p in per_rep]
        pops, errs, disagree, worst, from_replicas = {}, {}, False, -1.0, False
        for k in STATE_NAMES:
            values, errors = [r["populations"][k] for r in reps], [r["errors"][k] for r in reps]
            mean, err, dis = combine_replicas(values, errors)
            pops[k], errs[k], disagree = mean, err, disagree or dis
            if err > worst:
                within, between = error_parts(values, errors)
                worst, from_replicas = err, between > within
        fe = free_energies_kcal(pops, kt)
        top = max(pops, key=pops.get)
        fe_err = {k: (round(kt / KJ_PER_KCAL * math.sqrt((errs[k] / pops[k]) ** 2 + (errs[top] / pops[top]) ** 2), 2)
                      if pops[k] > 0 and k != top else (0.0 if k == top else None)) for k in STATE_NAMES}
        residues.append({
            "residue": label,
            "populations": {k: round(v, 3) for k, v in pops.items()},
            "standard_error": {k: round(v, 3) for k, v in errs.items()},
            "max_standard_error": round(max(errs.values()), 3),
            "free_energy_kcal": fe, "free_energy_error_kcal": fe_err,
            "per_replica": [{k: round(v, 3) for k, v in r["populations"].items()} for r in reps],
            "transitions_per_replica": [r["transitions"] for r in reps],
            "replicas_disagree": disagree,
            "error_from_replicas": from_replicas,
        })
    return {"ns_per_replica": round(min(ns), 6), "n_replicas": len(per_rep), "effective_samples": n_eff,
            "weights_degenerate": min(n_eff) < rule.min_effective_samples,
            "method": "Tiwary-Parrinello time-dependent reweighting; first "
                      f"{rule.skip_fraction:.0%} of each replica left out; errors from block averaging",
            "residues": residues}


def assess(result: dict, previous: dict | None, rule: StopRule, precision: float) -> dict[str, str]:
    """A verdict for each residue:

    reached     the goal is met: error at most `precision`, replicas agree, every replica crossed
                between states at least `min_transitions` times, weights not degenerate, long enough
    stalled     some replica hardly crosses between states: the bias does not push this side chain
                (a small error then means nothing: all replicas may be trapped in the same state)
    unreliable  the weights are degenerate, so the populations rest on a few frames
    not_yet     still too uncertain, or the replicas disagree

    The change since the previous check is only a diagnosis (see change_since_last), never a reason to stop.
    """
    prev = {r["residue"]: r["populations"] for r in (previous or {}).get("residues", [])}
    out = {}
    for r in result["residues"]:
        old = prev.get(r["residue"])
        r["change_since_last"] = (round(max(abs(r["populations"][k] - old[k]) for k in STATE_NAMES), 3)
                                  if old else None)
        if min(r["transitions_per_replica"]) < rule.min_transitions:
            out[r["residue"]] = "stalled"
        elif result["weights_degenerate"]:
            out[r["residue"]] = "unreliable"
        elif (r["max_standard_error"] <= precision and not r["replicas_disagree"]
              and result["ns_per_replica"] >= rule.min_ns - 1e-9):
            out[r["residue"]] = "reached"
        else:
            out[r["residue"]] = "not_yet"
        r["verdict"] = out[r["residue"]]
    return out


def format_table(result: dict) -> str:
    lines = [f"Reweighted chi1 populations after {result['ns_per_replica']:g} ns per replica "
             f"({result['n_replicas']} replicas; {result['method']}).",
             f"Effective samples per replica: {result['effective_samples']}"
             + (" (too few: the weights are degenerate)" if result["weights_degenerate"] else "") + ".",
             f"{'residue':<10}{'g+':>12}{'t':>12}{'g-':>12}   {'transitions':<14}{'verdict':<12}"]
    for r in result["residues"]:
        p, e = r["populations"], r["standard_error"]
        cells = "".join(f"{p[k]:>6.2f}±{e[k]:<5.2f}" for k in STATE_NAMES)
        trans = ",".join(map(str, r["transitions_per_replica"]))
        flag = " replicas disagree" if r["replicas_disagree"] else ""
        lines.append(f"{r['residue']:<10}{cells}   {trans:<14}{r.get('verdict', ''):<12}{flag}")
    return "\n".join(lines)


def save(run_dir: Path, result: dict) -> Path:
    path = Path(run_dir) / "enhanced" / "populations.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(path)
    return path
