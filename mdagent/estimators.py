"""Task-specific estimators for the general loop (mdagent.core).

Each estimator turns per-frame states of one estimand into an evidence.Assessment, with its own success
criteria and validity checks. The numbers come from the existing, tested code: block averaging and
error floors (uncertainty.py), the Markov state model (msm.py) and the first-entry rule (observables.py).

How frames become states (chi1 rotamers, near/far distances, ...) is the job of featurizers
(diagnose.assign_states, observables.two_state); estimators only see StateData.

Registered here (kind, regime -> estimator):
- state_populations, unbiased -> UnbiasedPopulations  (time fractions; block errors; error floor from
                                                       the number of state changes)
- state_populations, biased   -> WeightedPopulations  (reweighted fractions; Kish effective samples)
- state_populations, seeded   -> SegmentMSMPopulations (Markov state model of segments; bootstrap errors)
- event, unbiased and seeded  -> FirstEntryEvent      (entered a state and stayed; not valid on biased data,
                                                       where a bias can force any state)
- transition_rates, unbiased  -> TransitionRates      (rates between states; Poisson errors of the counts)
- free_energy_surface, unbiased -> FreeEnergySurface  (2D landscape of two angles; errors from replica spread)

Default criteria (min events per replica, effective samples) are estimator options that a Requirement can
override in its params. They are not global thresholds of the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core.evidence import Assessment, Check
from .core.registry import EstimatorRegistry
from .uncertainty import (block_standard_error, combine_replicas, effective_samples, error_parts,
                          floor_from_samples, floor_from_transitions)


@dataclass
class AngleData:
    """Two angles per frame (for example backbone phi/psi in degrees), one array each per replica, with the
    states of the same frames (for the number of independent events) and the time between frames."""
    phi: list
    psi: list
    states: StateData
    dt_ns: float


@dataclass
class StateData:
    states: list            # One int array per replica (or per segment for "seeded"); -1 = no state yet.
    names: list             # State names, index = state number.
    weights: list | None = None   # Biased regime: one weight array per replica.


def count_changes(states: np.ndarray) -> int:
    s = np.asarray(states)
    s = s[s >= 0]
    return int(np.count_nonzero(s[1:] != s[:-1]))


def _status(total_events: int, checks: list, precise: bool) -> str:
    if total_events == 0:
        return "indeterminate"
    return "met" if precise and all(c.passed for c in checks if c.blocking) else "not_met"


def _populations(data: StateData, weighted: bool):
    """Mean population of each state over replicas, its error, disagreement and the error parts of the worst state."""
    pops, errs, differ, parts, worst = {}, {}, False, (0.0, 0.0), -1.0
    for i, name in enumerate(data.names):
        values, errors = [], []
        for r, s in enumerate(data.states):
            s = np.asarray(s)
            ok = s >= 0
            hit = (s[ok] == i).astype(float)
            w = np.asarray(data.weights[r], dtype=float)[ok] if weighted else np.ones_like(hit)
            values.append(float(np.sum(w * hit) / np.sum(w)) if w.sum() > 0 else 0.0)
            errors.append(block_standard_error(hit, w if weighted else None))
        mean, err, dis = combine_replicas(values, errors)
        pops[name], errs[name], differ = mean, err, differ or dis
        if err > worst:
            worst, parts = err, error_parts(values, errors)
    return pops, errs, differ, parts


class UnbiasedPopulations:
    name, kind, regimes = "populations.unbiased_time_fraction", "state_populations", ("unbiased",)

    def assess(self, estimand: str, data: StateData, params: dict) -> Assessment:
        max_se, min_events = params["max_se"], params.get("min_events_per_replica", 4)
        per = [count_changes(s) for s in data.states]
        pops, errs, differ, (within, between) = _populations(data, weighted=False)
        errs = {k: max(v, floor_from_transitions(pops[k], sum(per), len(per))) for k, v in errs.items()}
        worst = max(errs.values())
        checks = [Check("events_in_every_replica", min(per) >= min_events, True,
                        f"state changes per replica {per}; needs {min_events} each"),
                  Check("replicas_consistent", not differ, True, "no two replicas differ by > 3 combined errors")]
        return Assessment(estimand, self.name, _status(sum(per), checks, worst <= max_se),
                          {k: round(v, 4) for k, v in pops.items()}, {k: round(v, 4) for k, v in errs.items()},
                          checks, {"events_total": sum(per), "events_per_replica": per,
                                   "events_needed": min_events * len(per), "uncertainty_ratio": round(worst / max_se, 4),
                                   "uncertainty_scaling": -0.5, "error_within": round(within, 5),
                                   "error_between": round(between, 5), "replicas_disagree": differ,
                                   "projectable": True, "regime": "unbiased"})


class WeightedPopulations:
    name, kind, regimes = "populations.reweighted", "state_populations", ("biased",)

    def assess(self, estimand: str, data: StateData, params: dict) -> Assessment:
        max_se = params["max_se"]
        min_events = params.get("min_events_per_replica_biased", 10)
        min_eff = params.get("min_effective_samples", 50.0)
        per = [count_changes(s) for s in data.states]
        n_eff = [effective_samples(np.asarray(w)[np.asarray(s) >= 0]) for s, w in zip(data.states, data.weights)]
        pops, errs, differ, (within, between) = _populations(data, weighted=True)
        errs = {k: max(v, floor_from_samples(pops[k], sum(n_eff))) for k, v in errs.items()}
        worst = max(errs.values())
        checks = [Check("events_in_every_replica", min(per) >= min_events, True,
                        f"state changes per replica {per}; needs {min_events} each under bias"),
                  Check("weights_not_degenerate", min(n_eff) >= min_eff, True,
                        f"Kish effective samples per replica {[round(x, 1) for x in n_eff]}; needs {min_eff:g}"),
                  Check("replicas_consistent", not differ, True, "no two replicas differ by > 3 combined errors")]
        return Assessment(estimand, self.name, _status(sum(per), checks, worst <= max_se),
                          {k: round(v, 4) for k, v in pops.items()}, {k: round(v, 4) for k, v in errs.items()},
                          checks, {"events_total": sum(per), "events_per_replica": per,
                                   "events_needed": min_events * len(per), "uncertainty_ratio": round(worst / max_se, 4),
                                   "uncertainty_scaling": -0.5, "error_within": round(within, 5),
                                   "error_between": round(between, 5), "replicas_disagree": differ,
                                   "projectable": True, "regime": "biased"})


class SegmentMSMPopulations:
    name, kind, regimes = "populations.markov_state_model", "state_populations", ("seeded",)

    def assess(self, estimand: str, data: StateData, params: dict) -> Assessment:
        from .msm import populations_with_errors

        max_se, min_total = params["max_se"], params.get("min_events_total", 12)
        pops, errs, left_out = populations_with_errors([np.asarray(s) for s in data.states], data.names)
        per = [count_changes(s) for s in data.states]
        worst = max(errs.values())
        checks = [Check("enough_events", sum(per) >= min_total, True, f"{sum(per)} state changes; needs {min_total}"),
                  Check("states_connected", not left_out, True,
                        f"states entered or left only one way: {left_out}" if left_out else "all states connected")]
        return Assessment(estimand, self.name, _status(sum(per), checks, worst <= max_se),
                          {k: round(v, 4) for k, v in pops.items()}, {k: round(v, 4) for k, v in errs.items()},
                          checks, {"events_total": sum(per), "events_per_replica": per, "events_needed": min_total,
                                   "uncertainty_ratio": round(worst / max_se, 4), "uncertainty_scaling": None,
                                   "projectable": False, "regime": "seeded"})


class FirstEntryEvent:
    name, kind, regimes = "event.first_entry_with_hold", "event", ("unbiased", "seeded")

    def assess(self, estimand: str, data: StateData, params: dict) -> Assessment:
        from .observables import first_entry

        target = data.names.index(params["target"])
        hold, seen_in = params.get("hold_frames", 1), params.get("seen_in", 1)
        seen = [first_entry(np.asarray(s), target, hold) for s in data.states]
        per = [count_changes(s) for s in data.states]
        n_seen = sum(f is not None for f in seen)
        checks = [Check("seen_in_enough_replicas", n_seen >= seen_in, True, f"seen in {n_seen}; needs {seen_in}")]
        status = "met" if n_seen >= seen_in else ("indeterminate" if sum(per) == 0 else "not_met")
        return Assessment(estimand, self.name, status, {"seen_in": n_seen, "first_frame": seen}, {}, checks,
                          {"events_total": sum(per), "events_per_replica": per, "events_needed": seen_in,
                           "projectable": False},
                          note="Seeing an event is evidence that it can happen; it gives no rate or probability.")


class TransitionRates:
    """Rates between states. Criterion: the largest relative error of the leading rates (between states with at
    least `min_population`) is at most `max_rel_error`. The relative error of a rate from N events is 1/sqrt(N)."""

    name, kind, regimes = "rates.poisson_counts", "transition_rates", ("unbiased",)

    def assess(self, estimand: str, data: StateData, params: dict) -> Assessment:
        from .backbone import transition_statistics, worst_rate_error

        max_rel, min_pop = params["max_rel_error"], params.get("min_population", 0.05)
        stats = transition_statistics(data.states, data.names, params.get("dt_ns", 1.0))
        total = sum(t for t in stats["time_ns"])
        pops = {a: (t / total if total else 0.0) for a, t in zip(data.names, stats["time_ns"])}
        worst, lead = worst_rate_error(stats, pops, min_pop)
        per = [count_changes(s) for s in data.states]
        need = int(np.ceil(1 / max_rel ** 2)) * max(len(lead), 1)
        finite = worst if np.isfinite(worst) else None
        checks = [Check("leading_rates_seen", finite is not None, True,
                        f"leading transitions {lead}" + ("" if finite is not None else "; at least one never seen"))]
        status = _status(sum(per), checks, finite is not None and worst <= max_rel)
        return Assessment(estimand, self.name, status, {"rates_per_ns": stats["rates_per_ns"], "leading": lead},
                          {"worst_relative_error": None if finite is None else round(worst, 4)}, checks,
                          {"events_total": sum(per), "events_per_replica": per, "events_needed": need,
                           "uncertainty_ratio": None if finite is None else round(worst / max_rel, 4),
                           "uncertainty_scaling": -0.5, "projectable": True, "regime": "unbiased"},
                          note="Poisson errors assume Markovian jumps between state cores.")


class FreeEnergySurface:
    """A 2D free-energy landscape of two angles. Criterion: the largest error of F in the bins with F <= `cutoff_kcal`
    is at most `max_error_kcal`. Needs at least two replicas (the error comes from their spread)."""

    name, kind, regimes = "landscape.replica_spread", "free_energy_surface", ("unbiased",)

    def assess(self, estimand: str, data: AngleData, params: dict) -> Assessment:
        from .backbone import landscape, worst_landscape_error

        max_err = params["max_error_kcal"]
        per = [count_changes(s) for s in data.states.states]
        fes = landscape(data.phi, data.psi, params.get("temperature_k", 300.0), params.get("bins_deg", 15.0),
                        n_effective=sum(per) + len(per))
        worst = worst_landscape_error(fes, params.get("cutoff_kcal", 2.0))
        finite = worst if np.isfinite(worst) else None
        populated = int(np.sum(np.isfinite(fes["free_energy_kcal"]) & (fes["free_energy_kcal"] <= params.get("cutoff_kcal", 2.0))))
        checks = [Check("two_or_more_replicas", len(per) >= 2, True, f"{len(per)} replicas; the error needs their spread")]
        status = _status(sum(per), checks, finite is not None and worst <= max_err)
        return Assessment(estimand, self.name, status, {"bins_with_low_free_energy": populated},
                          {"worst_error_kcal": None if finite is None else round(worst, 4)}, checks,
                          {"events_total": sum(per), "events_per_replica": per,
                           "uncertainty_ratio": None if finite is None else round(worst / max_err, 4),
                           "uncertainty_scaling": -0.5, "projectable": True, "regime": "unbiased"},
                          note="Errors from the spread between replicas, never below an error floor from the number "
                               "of state changes.")


def default_registry() -> EstimatorRegistry:
    return EstimatorRegistry([UnbiasedPopulations(), WeightedPopulations(), SegmentMSMPopulations(), FirstEntryEvent(),
                              TransitionRates(), FreeEnergySurface()])
