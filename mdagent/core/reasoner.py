"""Competing hypotheses about why an estimand is not resolved yet.

The reasoner reads only the shared evidence vocabulary (evidence.SIGNALS) and the progress of the run.
It does not know what is measured. Every hypothesis is a heuristic argument, not a statistical result:
`basis` says so, and the ledger and the report keep that label.

Hypotheses (one estimand can have several, and they compete):
- slow_mixing: events happen in every replica, but too few or the error is too large. More data of the
  same kind should help.
- replica_dependence: replicas disagree beyond their errors: the result depends on the starting point.
  More independent replicas (or restarts from chosen frames) help more than longer ones.
- partial_trapping: some replicas show events and others none.
- trapped_or_single_state: no event in any replica. A kinetic trap and a single stable state cannot be
  told apart from these data; only a method that pushes the system can tell.
- not_yet_observed: the competing explanation for zero events while the remaining budget could still show
  some (only when trapped_or_single_state is not "high").
- unexplained: not resolved, and no signal points to a cause.
- estimator_invalid: a blocking validity check failed for a reason other than too few events (for
  example degenerate reweighting weights or a disconnected state model). The estimate is not usable.
- bias_ineffective: biased data, yet hardly any events: the bias does not push the slow motion, which is
  probably coupled to something else.
- requirement_unreachable: even an optimistic extrapolation of the rate seen so far needs more than the
  largest budget the agent may use.

Trap plausibility with zero events uses the 95 % upper bound of a Poisson rate ("rule of three": 3 / exposure).
If even that rate cannot deliver the events the estimator needs within the remaining budget, more of the
same is very unlikely to help (plausibility high). This replaces a fixed patience time, which would be
system-specific.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PLAUSIBILITY_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}
UNREACHABLE_MARGIN = 1.5   # Only call a requirement unreachable when even the optimistic estimate is this much over.
MIN_LOOKS_FOR_PROJECTION = 2


@dataclass
class Hypothesis:
    id: str
    estimand: str
    plausibility: str            # high, medium, low
    support: list = field(default_factory=list)   # Evidence statements.
    projected_cost: float | None = None           # Optimistic total cost to meet the requirement, if estimated.
    basis: str = "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Progress:
    """Where the run is. Cost in budget units; length per replica of the current tool."""
    tool: str
    length: float
    n_replicas: int
    spent: float
    looks: int


def hypothesize(assessments: list, progress: Progress, max_cost: float, cost_factor: float = 1.0,
                previous_projection: dict | None = None) -> list[Hypothesis]:
    """`previous_projection`: {estimand: projected cost at the previous look}. A requirement is called
    unreachable only when the previous and the current projection are both over the margin (uncertainty
    estimates are noisy from look to look, so one pessimistic look must not stop a run)."""
    previous_projection = previous_projection or {}
    out: list[Hypothesis] = []
    exposure = progress.length * progress.n_replicas          # Replica-length of the current tool.
    remaining = max(max_cost - progress.spent, 0.0) / cost_factor
    for a in assessments:
        if a.passes:
            continue
        s = a.signals
        events = s.get("events_total")
        per = s.get("events_per_replica") or []
        needed = s.get("events_needed")
        regime = s.get("regime", "unbiased")
        failed = [c for c in a.checks if c.blocking and not c.passed]

        if events == 0:
            if regime == "biased":
                out.append(Hypothesis("bias_ineffective", a.estimand, "high",
                                      ["biased data, yet no event in any replica"]))
            else:
                plaus, why = "medium", "no event in any replica"
                if exposure > 0 and needed:
                    upper = 3.0 / exposure * remaining
                    why += f"; even at the 95 % upper rate, the remaining budget gives about {upper:.2g} events"
                    if upper < needed:
                        plaus = "high"
                        why += f", fewer than the {needed} needed"
                out.append(Hypothesis("trapped_or_single_state", a.estimand, plaus, [why]))
                if plaus != "high":   # The competing explanation: events are slow, but the budget can still show them.
                    out.append(Hypothesis("not_yet_observed", a.estimand, "medium",
                                          ["no event yet, but the remaining budget could still show some"]))
            continue

        if per and min(per) == 0 and max(per) > 0:
            out.append(Hypothesis("partial_trapping", a.estimand, "high" if regime != "biased" else "medium",
                                  [f"events per replica {per}"]))
            if regime == "biased":
                out.append(Hypothesis("bias_ineffective", a.estimand, "medium",
                                      [f"biased data, but some replicas show no event: {per}"]))
        if s.get("replicas_disagree"):
            out.append(Hypothesis("replica_dependence", a.estimand, "high",
                                  ["replicas differ beyond their errors"]))
        elif (s.get("error_between") or 0) > (s.get("error_within") or 0) > 0:
            out.append(Hypothesis("replica_dependence", a.estimand, "low",
                                  ["the spread between replicas is the larger error part"]))
        other_failures = [c for c in failed if c.name not in ("enough_events", "events_in_every_replica")]
        if other_failures:
            out.append(Hypothesis("estimator_invalid", a.estimand, "high",
                                  [f"failed check {c.name}: {c.detail}" for c in other_failures]))
        if events and (not per or min(per) > 0):
            out.append(Hypothesis("slow_mixing", a.estimand, "medium",
                                  [f"{events} events so far; uncertainty ratio {s.get('uncertainty_ratio')}"]))

        projected = _projected_cost(a, progress, cost_factor)
        if projected is not None:
            for h in out:
                if h.estimand == a.estimand and h.id == "slow_mixing":
                    h.projected_cost = projected
            before = previous_projection.get(a.estimand)
            if before is not None and min(before, projected) > UNREACHABLE_MARGIN * max_cost:
                best = min(before, projected)
                out.append(Hypothesis("requirement_unreachable", a.estimand, "medium",
                                      [f"optimistic extrapolation needs about {best:.3g} budget units at two "
                                       f"looks in a row, more than {UNREACHABLE_MARGIN:g} x the limit {max_cost:g}"],
                                      projected_cost=best))
        if not any(h.estimand == a.estimand for h in out):
            out.append(Hypothesis("unexplained", a.estimand, "low", [f"status {a.status}; no specific signal"]))
    return out


def _projected_cost(a, progress: Progress, cost_factor: float) -> float | None:
    """Optimistic total cost (spent + more) to meet the requirement, from the rate seen so far."""
    s = a.signals
    if not s.get("projectable") or progress.looks < MIN_LOOKS_FOR_PROJECTION or progress.length <= 0:
        return None
    events, needed = s.get("events_total") or 0, s.get("events_needed")
    exposure = progress.length * progress.n_replicas
    need = exposure
    if needed and events > 0 and events < needed:
        need = max(need, exposure * needed / events)
    ratio, scaling = s.get("uncertainty_ratio"), s.get("uncertainty_scaling")
    if ratio and scaling and ratio > 1:
        need = max(need, exposure * ratio ** (1.0 / abs(scaling)))
    spent_here = exposure * cost_factor
    return progress.spent - spent_here + need * cost_factor
