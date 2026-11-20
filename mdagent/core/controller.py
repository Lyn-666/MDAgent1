"""Budget- and permission-constrained choice of the next action.

Candidate actions (general verbs; the tool decides what they mean physically):
- continue       run every replica one more step within the planned budget (the user's own plan:
                 no permission needed)
- extend_budget  the same beyond the planned budget, up to the hard limit (permission "extend")
- add_replicas   start more independent replicas, which first catch up (permission "add_replicas" when
                 it goes past the planned replicas)
- switch_method  continue with another tool that serves the estimand (permission "switch_method",
                 auto mode, and the tool listed in TaskSpec.allowed_methods)
- stop           stopping before the planned budget is used needs "stop_early"; at the end of the
                 budget, stopping is always allowed

Value: each candidate gets a heuristic expected information value: for every open estimand, the
strongest of its hypotheses (reasoner.py) times how much this kind of action addresses that hypothesis
(EFFECT). This is a ranking device, not a probability. Ties go to the cheaper action, then to the order
of KINDS, so the same evidence always gives the same choice.

Selection:
1. Every required estimand resolved: stop with "target_reached" (before the planned end only with
   stop_early; otherwise the plan goes on and the record says the goal is met).
2. Otherwise the authorized candidate with the highest value > 0.
3. No authorized candidate has value: follow the plan if there is plan left and stopping early is not
   allowed; otherwise stop. The stop reason is "stalled" (the open estimands look trapped or the bias
   does not push them), "not_reachable" (even an optimistic extrapolation is over the limit) or
   "budget_exhausted".
Unauthorized candidates that would be worth more than the choice become suggestions, with what the user
would have to allow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .reasoner import PLAUSIBILITY_WEIGHT, Hypothesis, Progress
from .task import TaskSpec

KINDS = ("continue", "extend_budget", "add_replicas", "switch_method", "stop")
STOP_REASONS = ("target_reached", "stalled", "not_reachable", "budget_exhausted", "failed")
EPS = 1e-9

# How much an action addresses a hypothesis (0 to 1). A heuristic, documented and tested; not fitted.
EFFECT = {
    "slow_mixing":             {"continue": 1.0, "extend_budget": 1.0, "add_replicas": 0.7, "switch_method": 0.4},
    "replica_dependence":      {"continue": 0.3, "extend_budget": 0.3, "add_replicas": 1.0, "switch_method": 0.6},
    "partial_trapping":        {"continue": 0.3, "extend_budget": 0.3, "add_replicas": 0.6, "switch_method": 0.8},
    "trapped_or_single_state": {"continue": 0.0, "extend_budget": 0.0, "add_replicas": 0.2, "switch_method": 1.0},
    "not_yet_observed":        {"continue": 1.0, "extend_budget": 1.0, "add_replicas": 0.5, "switch_method": 0.3},
    "estimator_invalid":       {"continue": 0.6, "extend_budget": 0.6, "add_replicas": 0.3, "switch_method": 0.4},
    "bias_ineffective":        {"continue": 0.0, "extend_budget": 0.0, "add_replicas": 0.1, "switch_method": 0.6},
    "unexplained":             {"continue": 0.5, "extend_budget": 0.5, "add_replicas": 0.3, "switch_method": 0.1},
    "awaiting_confirmation":   {"continue": 1.0, "extend_budget": 1.0, "add_replicas": 0.0, "switch_method": 0.0},
}
STALLING = ("trapped_or_single_state", "bias_ineffective")


@dataclass
class Candidate:
    kind: str
    tool: str
    amount: float = 0.0        # Length per replica to add, or number of replicas to add.
    cost: float = 0.0          # Budget units.
    value: float = 0.0
    permission: str | None = None
    addresses: list = field(default_factory=list)   # [estimand, hypothesis id]
    authorized: bool = False
    blocked_by: str = ""       # Why it is not authorized.

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    chosen: Candidate
    stop_reason: str | None
    rationale: str
    open_estimands: list
    candidates: list
    suggestions: list
    unsupported: list = field(default_factory=list)   # Open estimands no registered tool can push.

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _value(kind: str, hyps: list[Hypothesis], covered: set | None = None) -> tuple[float, list]:
    """Sum over estimands of the strongest (plausibility x effect). `covered`: estimands the action can serve."""
    best: dict[str, tuple[float, str]] = {}
    unreachable = {h.estimand for h in hyps if h.id == "requirement_unreachable"}
    for h in hyps:
        if covered is not None and h.estimand not in covered:
            continue
        effect = EFFECT.get(h.id, {}).get(kind, 0.0)
        if kind in ("continue", "extend_budget") and h.estimand in unreachable:
            effect = 0.0   # More of the same cannot reach this requirement within the limit.
        v = PLAUSIBILITY_WEIGHT[h.plausibility] * effect
        if v > best.get(h.estimand, (0.0, ""))[0]:
            best[h.estimand] = (v, h.id)
    return round(sum(v for v, _ in best.values()), 6), sorted([e, hid] for e, (v, hid) in best.items())


def propose(task: TaskSpec, hyps: list[Hypothesis], progress: Progress, tools, estimators, open_kinds: dict) -> list[Candidate]:
    """All candidate actions with cost, value and authorization. `open_kinds`: {open estimand: kind}."""
    granted, b = task.granted(), task.budget
    tool = tools.get(progress.tool)
    n, cf = progress.n_replicas, tool.cost_factor
    cap = b.max_cost if "extend" in granted else b.planned_cost
    out: list[Candidate] = []

    plan_left = (b.planned_cost - progress.spent) / (n * cf)
    if plan_left > EPS:
        amount = min(task.step, plan_left)
        out.append(Candidate("continue", tool.name, amount, amount * n * cf))
    else:   # Always proposed, so that it can become a suggestion even when the hard limit is reached.
        amount = min(task.step, (b.max_cost - progress.spent) / (n * cf))
        amount = amount if amount > EPS else task.step
        out.append(Candidate("extend_budget", tool.name, amount, amount * n * cf, permission="extend"))

    k = 2 if n >= 2 else 1   # Two new replicas at a time (one for a single replica).
    out.append(Candidate("add_replicas", tool.name, k, k * progress.length * cf,
                         permission="add_replicas" if n + k > b.planned_replicas else None))

    for name in tools.names():
        if name == tool.name:
            continue
        other = tools.get(name)
        covered = {e for e, kind in open_kinds.items()
                   if other.supports(kind) and estimators.find(kind, other.regime) is not None}
        if covered and other.regime != tool.regime:
            c = Candidate("switch_method", name, task.step, task.step * n * other.cost_factor,
                          permission="switch_method")
            c.value, c.addresses = _value("switch_method", hyps, covered)
            out.append(c)

    for c in out:
        if c.kind != "switch_method":
            c.value, c.addresses = _value(c.kind, hyps)
        why = []
        if c.permission and c.permission not in granted:
            why.append(f"needs permission {c.permission}")
        if c.kind == "switch_method" and c.tool not in task.allowed_methods:
            why.append(f"method {c.tool} is not among the allowed methods")
        if c.kind == "add_replicas" and n + c.amount > (b.max_replicas if "add_replicas" in granted else b.planned_replicas):
            why.append(f"would pass the replica limit {b.max_replicas}")
        limit = b.planned_cost if c.kind == "continue" else cap
        if progress.spent + c.cost > limit + EPS:
            why.append(f"would pass the budget {limit:g}")
        c.blocked_by = "; ".join(why)
        c.authorized = not why
    return out


def select(task: TaskSpec, cands: list[Candidate], hyps: list[Hypothesis], progress: Progress,
           open_estimands: list, unsupported: list) -> Decision:
    granted, b = task.granted(), task.budget
    plan = next((c for c in cands if c.kind == "continue" and c.authorized), None)

    def rank(c):
        return (-c.value, c.cost, KINDS.index(c.kind), c.tool)

    def finish(chosen, reason, why):
        better = sorted([c for c in cands if not c.authorized and c.value > max(chosen.value, 0.0)], key=rank)
        return Decision(chosen, reason, why, list(open_estimands), cands, better, list(unsupported))

    if not open_estimands:
        if plan and "stop_early" not in granted:
            return finish(plan, None, "Every required estimand is resolved, but stopping early is not allowed: "
                                      "the plan goes on.")
        return finish(Candidate("stop", progress.tool, authorized=True), "target_reached",
                      "Every required estimand is resolved.")

    useful = sorted([c for c in cands if c.authorized and c.value > 0], key=rank)
    if useful:
        c = useful[0]
        return finish(c, None, f"Highest expected information value among allowed actions ({c.value:g}); "
                               f"addresses {c.addresses}.")
    if plan and "stop_early" not in granted:
        return finish(plan, None, "No allowed action is expected to help, but stopping early is not allowed: "
                                  "the plan goes on.")

    ids = {h.id for h in hyps if h.estimand in open_estimands and (h.id not in ("trapped_or_single_state",)
                                                                  or h.plausibility == "high")}
    if ids & set(STALLING):
        reason = "stalled"
    elif "requirement_unreachable" in ids:
        reason = "not_reachable"
    elif plan:
        reason = "stalled"
    else:
        reason = "budget_exhausted"
    early = " before the end of the planned budget" if plan else ""
    return finish(Candidate("stop", progress.tool, authorized=True), reason,
                  f"No allowed action is expected to help{early}: {reason}.")
