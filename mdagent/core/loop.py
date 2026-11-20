"""The general decision loop.

At every decision point:
1. estimands and success requirements: from the TaskSpec;
2. evidence: each estimand is assessed by the estimator registered for its kind and the data regime of
   the newest tool that serves it;
3./4. unresolved uncertainty and competing hypotheses: reasoner.hypothesize;
5./6. candidate actions with value and cost, choice under permissions and budget: controller;
7. only an authorized action is executed; the budget and replica limits are checked again afterwards,
   and a violation stops the run with an error (it means a tool did more than it was asked);
8. the decision and its evidence are written to the ledger before the action runs.

Resuming: the loop reads the ledger and the tools' own progress, so a stopped loop continues where it
was. A loop whose last decision stopped it does not run again; after a failure (an error, or the step
limit) it can be resumed once the cause is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .controller import EPS, propose, select
from .evidence import Assessment
from .ledger import Ledger
from .reasoner import Hypothesis, Progress, hypothesize
from .registry import EstimatorRegistry, ToolRegistry
from .task import TaskSpec


class PermissionViolation(RuntimeError):
    pass


class BudgetViolation(RuntimeError):
    pass


@dataclass
class Outcome:
    stop_reason: str
    spent: float
    looks: int
    history: list
    assessments: dict = field(default_factory=dict)   # estimand -> Assessment as dict
    suggestions: list = field(default_factory=list)


def spent(tools: ToolRegistry, history: list) -> float:
    return sum(t.length() * t.replicas() * t.cost_factor for t in (tools.get(n) for n in dict.fromkeys(history)))


def _source(kind: str, history: list, tools: ToolRegistry, estimators: EstimatorRegistry):
    """The newest tool of the run that serves this kind and has data, with its estimator."""
    for name in reversed(history):
        t = tools.get(name)
        e = estimators.find(kind, t.regime)
        if t.supports(kind) and e is not None and t.length() > EPS:
            return t, e
    return None, None


def _limits(task: TaskSpec) -> tuple[float, int]:
    g, b = task.granted(), task.budget
    return (b.max_cost if "extend" in g else b.planned_cost,
            b.max_replicas if "add_replicas" in g else b.planned_replicas)


def _check_limits(task: TaskSpec, tools: ToolRegistry, history: list) -> float:
    cost_cap, replica_cap = _limits(task)
    now = spent(tools, history)
    if now > cost_cap + 1e-6:
        raise BudgetViolation(f"spent {now:g} > limit {cost_cap:g} {task.budget.unit}")
    for name in history:
        if tools.get(name).replicas() > replica_cap:
            raise BudgetViolation(f"tool {name} runs {tools.get(name).replicas()} replicas > limit {replica_cap}")
    return now


def run(task: TaskSpec, tools: ToolRegistry, estimators: EstimatorRegistry, ledger: Ledger,
        max_steps: int = 1000) -> Outcome:
    last = ledger.last
    if last and last["kind"] == "decision" and last.get("stop_reason"):   # Finished. A failure can be resumed.
        return Outcome(last["stop_reason"], last["progress"]["spent"], last["progress"]["looks"],
                       last["progress"]["history"], {a["estimand"]: a for a in last.get("assessments", [])},
                       last.get("decision", {}).get("suggestions", []))
    history = list(last["progress"]["history"]) if last else [task.method]
    streak = next((dict(r["streak"]) for r in reversed(ledger.records) if "streak" in r), {})
    looks = last["progress"]["looks"] if last else 0
    tool = tools.get(history[-1])

    if tool.length() <= EPS:   # The first round of the user's plan.
        n = task.budget.planned_replicas
        amount = min(task.step, task.budget.planned_cost / (n * tool.cost_factor))
        ledger.append({"kind": "start", "progress": _progress(tool, 0.0, looks, history),
                       "executed": {"kind": "continue", "tool": tool.name, "amount": amount}})
        tool.advance(n, amount)
        _check_limits(task, tools, history)

    for _ in range(max_steps):
        now = _check_limits(task, tools, history)
        assessments = []
        for r in task.requirements:
            t, est = _source(r.kind, history, tools, estimators)
            if t is None:
                assessments.append(Assessment(r.estimand, "none", "indeterminate",
                                              note="no tool used in this run serves this estimand"))
                continue
            assessments.append(est.assess(r.estimand, t.observe([r.estimand])[r.estimand], r.params))
        looks += 1
        for a in assessments:
            streak[a.estimand] = streak.get(a.estimand, 0) + 1 if a.passes else 0
        reqs = {r.estimand: r for r in task.requirements}
        open_ = [e for e, r in reqs.items() if r.required and streak[e] < r.confirm_looks]
        progress = Progress(tool.name, tool.length(), tool.replicas(), now, looks)

        before = next((r for r in reversed(ledger.records) if r["kind"] == "decision"), None)
        previous = {h["estimand"]: h["projected_cost"] for h in (before or {}).get("hypotheses", [])
                    if h["id"] == "slow_mixing" and h.get("projected_cost") is not None
                    and before["progress"]["tool"] == tool.name}
        hyps = hypothesize([a for a in assessments if a.estimand in open_], progress,
                           _limits(task)[0], tool.cost_factor, previous)
        hyps += [Hypothesis("awaiting_confirmation", e, "high",
                            [f"met at {streak[e]} of {reqs[e].confirm_looks} consecutive looks"])
                 for e in open_ if streak[e] > 0]
        open_kinds = {e: reqs[e].kind for e in open_}
        unsupported = [e for e in open_ if any(h.estimand == e and h.id in ("trapped_or_single_state", "bias_ineffective")
                                                for h in hyps)
                       and not any(n != tool.name and tools.get(n).supports(open_kinds[e])
                                   and estimators.find(open_kinds[e], tools.get(n).regime) is not None
                                   and tools.get(n).regime != tool.regime for n in tools.names())]
        cands = propose(task, hyps, progress, tools, estimators, open_kinds)
        decision = select(task, cands, hyps, progress, open_, unsupported)
        c = decision.chosen
        ledger.append({"kind": "decision", "progress": _progress(tool, now, looks, history), "streak": streak,
                       "task": task.to_dict() if looks == 1 else None,
                       "assessments": [a.to_dict() for a in assessments],
                       "hypotheses": [h.to_dict() for h in hyps], "decision": decision.to_dict(),
                       "stop_reason": decision.stop_reason})
        if not c.authorized:
            raise PermissionViolation(f"the controller chose an unauthorized action: {c.kind} ({c.blocked_by})")
        if c.kind == "stop":
            return Outcome(decision.stop_reason, now, looks, history, {a.estimand: a.to_dict() for a in assessments},
                           [s.to_dict() for s in decision.suggestions])
        try:
            if c.kind in ("continue", "extend_budget"):
                tool.advance(tool.replicas(), tool.length() + c.amount)
            elif c.kind == "add_replicas":
                tool.advance(tool.replicas() + int(c.amount), tool.length())
            elif c.kind == "switch_method":
                new = tools.get(c.tool)
                history.append(c.tool)
                new.advance(tool.replicas(), new.length() + c.amount)
                tool = new
            else:
                raise ValueError(f"unknown action {c.kind}")
            _check_limits(task, tools, history)
        except Exception as e:
            ledger.append({"kind": "failure", "progress": _progress(tool, spent(tools, history), looks, history),
                           "error": f"{type(e).__name__}: {e}", "stop_reason": "failed"})
            raise
    ledger.append({"kind": "failure", "progress": _progress(tool, spent(tools, history), looks, history),
                   "error": f"no stop after {max_steps} decisions", "stop_reason": "failed"})
    return Outcome("failed", spent(tools, history), looks, history)


def _progress(tool, now: float, looks: int, history: list) -> dict:
    return {"tool": tool.name, "length": tool.length(), "n_replicas": tool.replicas(), "spent": now,
            "looks": looks, "history": list(history)}
