"""The general decision loop on synthetic systems: execution, permissions, budget, stop reasons, records.

Every test is deterministic: synthetic data come from seeded generators and the ledger clock is fixed.
"""

import itertools
import json

import pytest

from mdagent.core.controller import KINDS
from mdagent.core.ledger import Ledger
from mdagent.core.loop import BudgetViolation, run
from mdagent.core.registry import ToolRegistry
from mdagent.core.task import PERMISSIONS, Budget, Requirement, TaskSpec
from mdagent.estimators import default_registry
from mdagent.synthetic import SyntheticEstimand, SyntheticTool, two_state

NAMES = ["a", "b"]
FAST = two_state(0.95, 0.9)
SLOW_FLIPS = two_state(0.995, 0.99)
TRAPPED = two_state(1.0, 1.0)
RARE = two_state(0.99999, 0.99998)
BIAS = two_state(0.9, 0.85)

SYSTEMS = {"fast": FAST, "slow": SLOW_FLIPS, "trapped": TRAPPED, "rare": RARE}


def _setup(P, mode="auto", permissions=None, max_se=0.03, cost=60.0, replicas=3, hard_cost=None,
           hard_replicas=None, start=None, biased_tool=True, seed=1, overrun=0.0, path=None):
    est = {"x": SyntheticEstimand("state_populations", P, NAMES, start=start, biased=BIAS)}
    tools = [SyntheticTool("md", est, seed=seed, overrun=overrun)]
    if biased_tool:
        tools.append(SyntheticTool("biased", est, regime="biased", seed=seed + 100))
    task = TaskSpec("populations of x", (Requirement("x", "state_populations", {"max_se": max_se}),),
                    Budget(cost, replicas, hard_cost, hard_replicas), "md", mode, permissions,
                    ("biased",), step=2.0)
    return task, ToolRegistry(tools), Ledger(path, clock=lambda: "T")


def _go(*a, **k):
    task, tools, ledger = _setup(*a, **k)
    return run(task, tools, default_registry(), ledger), ledger, task, tools


def _decisions(ledger):
    return [r for r in ledger.records if r["kind"] == "decision"]


# ---------------------------------------------------------------- decision safety over many cases

def _all_permission_sets():
    for k in range(len(PERMISSIONS) + 1):
        yield from itertools.combinations(PERMISSIONS, k)


@pytest.mark.parametrize("system", sorted(SYSTEMS))
@pytest.mark.parametrize("mode", ["auto", "guided"])
def test_safety_invariants_hold_for_every_permission_set(system, mode):
    for perms in _all_permission_sets():
        out, ledger, task, tools = _go(SYSTEMS[system], mode, perms, hard_cost=90.0, hard_replicas=5)
        granted = task.granted()
        cap = 90.0 if "extend" in granted else 60.0
        rep_cap = 5 if "add_replicas" in granted else 3
        assert out.stop_reason in ("target_reached", "stalled", "not_reachable", "budget_exhausted")
        assert out.spent <= cap + 1e-6
        for r in _decisions(ledger):
            c = r["decision"]["chosen"]
            assert c["authorized"] and c["kind"] in KINDS
            assert c["permission"] is None or c["permission"] in granted
            assert r["progress"]["n_replicas"] <= rep_cap
            if mode == "guided":
                assert c["kind"] != "switch_method"
            for s in r["decision"]["suggestions"]:
                assert not s["authorized"]
        final = _decisions(ledger)[-1]
        if out.stop_reason == "target_reached":
            assert all(a["status"] == "met" and all(c["passed"] for c in a["checks"] if c["blocking"])
                       for a in final["assessments"])
        if "stop_early" not in granted and out.stop_reason != "target_reached":
            assert out.spent >= 60.0 - 1e-6, "stopped before the plan's end without permission"


def test_trapped_system_is_never_reported_as_converged():
    for seed in range(5):
        out, _, _, _ = _go(TRAPPED, "auto", PERMISSIONS, seed=seed, start=[0, 1, 0], biased_tool=False)
        assert out.stop_reason != "target_reached"
        assert out.assessments["x"]["status"] == "indeterminate"


# ---------------------------------------------------------------- modes and permissions

def test_auto_default_stops_early_once_the_goal_is_met_and_saves_budget():
    out, _, _, _ = _go(FAST)
    assert out.stop_reason == "target_reached" and out.spent < 60.0


def test_without_stop_early_the_plan_runs_to_its_end_and_the_record_says_why():
    out, ledger, _, _ = _go(FAST, "auto", ())
    assert out.stop_reason == "target_reached" and out.spent == pytest.approx(60.0)
    assert any("stopping early is not allowed" in r["decision"]["rationale"] for r in _decisions(ledger))


def test_trapped_auto_default_stops_as_stalled_and_only_suggests_switching():
    out, ledger, _, tools = _go(RARE)
    assert out.stop_reason == "stalled" and out.history == ["md"]
    assert tools.get("biased").length() == 0
    assert any(s["kind"] == "switch_method" and "switch_method" in s["blocked_by"] for s in out.suggestions)


def test_with_permission_auto_switches_and_the_reweighted_result_is_right():
    out, _, _, _ = _go(RARE, "auto", ("stop_early", "switch_method"))
    assert out.history == ["md", "biased"] and out.stop_reason == "target_reached"
    a = out.assessments["x"]
    assert a["estimator"] == "populations.reweighted"
    assert abs(a["estimate"]["a"] - 2 / 3) <= 3 * a["uncertainty"]["a"]


def test_guided_never_switches_even_with_the_permission():
    out, _, _, tools = _go(RARE, "guided", PERMISSIONS)
    assert out.history == ["md"] and tools.get("biased").length() == 0


def test_a_switch_target_must_be_listed_as_allowed():
    task, tools, ledger = _setup(RARE, "auto", ("stop_early", "switch_method"))
    task = TaskSpec(task.question, task.requirements, task.budget, "md", "auto", task.permissions, (), task.step)
    out = run(task, tools, default_registry(), ledger)
    assert out.history == ["md"]
    assert any("not among the allowed methods" in s["blocked_by"] for s in out.suggestions)


def test_replicas_are_added_only_with_permission_and_up_to_the_hard_limit():
    out, ledger, _, tools = _go(TRAPPED, "auto", ("stop_early", "add_replicas"), start=[0, 1], replicas=2,
                                hard_replicas=4, biased_tool=False)
    assert tools.get("md").replicas() == 4
    assert any(r["decision"]["chosen"]["kind"] == "add_replicas" for r in _decisions(ledger))
    out, _, _, tools = _go(TRAPPED, "auto", ("stop_early",), start=[0, 1], replicas=2, hard_replicas=4,
                           biased_tool=False)
    assert tools.get("md").replicas() == 2


# ---------------------------------------------------------------- stop reasons

def test_not_reachable_when_even_an_optimistic_projection_is_over_the_limit():
    out, ledger, _, _ = _go(SLOW_FLIPS, max_se=0.002)
    assert out.stop_reason == "not_reachable" and out.spent < 60.0
    hyp = [h for h in _decisions(ledger)[-1]["hypotheses"] if h["id"] == "requirement_unreachable"]
    assert hyp and hyp[0]["basis"] == "heuristic" and hyp[0]["projected_cost"] > 1.5 * 60


def test_budget_exhausted_suggests_an_extension_and_extends_only_with_permission():
    # A precision just out of reach of this budget (seed 1): reachable by extrapolation, but not in 60.
    out, _, _, _ = _go(FAST, max_se=0.022)
    assert out.stop_reason == "budget_exhausted" and out.spent == pytest.approx(60.0)
    assert any(s["kind"] == "extend_budget" for s in out.suggestions)
    out, _, _, _ = _go(FAST, "auto", ("stop_early", "extend"), max_se=0.022, hard_cost=120.0)
    assert out.spent > 60.0 and out.spent <= 120.0 + 1e-6


def test_repeated_look_guard_needs_consecutive_passes():
    task, tools, ledger = _setup(FAST)
    req = Requirement("x", "state_populations", {"max_se": 0.03}, confirm_looks=2)
    task = TaskSpec(task.question, (req,), task.budget, "md", "auto", None, ("biased",), 2.0)
    out = run(task, tools, default_registry(), ledger)
    passes = [all(a["status"] == "met" for a in r["assessments"]) for r in _decisions(ledger)]
    assert out.stop_reason == "target_reached" and passes[-2:] == [True, True]


# ---------------------------------------------------------------- execution, records, resume

def test_same_inputs_give_the_same_records():
    _, a, _, _ = _go(RARE, "auto", ("stop_early", "switch_method"))
    _, b, _, _ = _go(RARE, "auto", ("stop_early", "switch_method"))
    assert json.dumps(a.records, sort_keys=True) == json.dumps(b.records, sort_keys=True)


def test_every_decision_record_holds_evidence_hypotheses_candidates_and_choice(tmp_path):
    _, ledger, _, _ = _go(SLOW_FLIPS, path=tmp_path / "decisions.jsonl")
    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == len(ledger.records)
    for r in _decisions(ledger):
        assert {"assessments", "hypotheses", "decision", "progress", "streak"} <= set(r)
        assert r["decision"]["candidates"] and r["decision"]["rationale"]
        assert all(h["basis"] == "heuristic" for h in r["hypotheses"])


def test_a_tool_that_overruns_its_budget_is_stopped_with_an_error():
    task, tools, ledger = _setup(FAST, overrun=30.0, biased_tool=False)
    with pytest.raises(BudgetViolation):
        run(task, tools, default_registry(), ledger)


def test_a_failed_run_resumes_and_ends_like_an_uninterrupted_one():
    reference, _, _, _ = _go(FAST)

    task, tools, ledger = _setup(FAST)
    md = tools.get("md")
    real, calls = md.advance, {"n": 0}

    def flaky(n, until):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash")
        real(n, until)
    md.advance = flaky
    with pytest.raises(RuntimeError):
        run(task, tools, default_registry(), ledger)
    assert ledger.last["kind"] == "failure" and ledger.last["stop_reason"] == "failed"
    out = run(task, tools, default_registry(), ledger)
    assert out.stop_reason == reference.stop_reason
    assert out.spent == pytest.approx(reference.spent)
    assert out.assessments["x"]["estimate"] == reference.assessments["x"]["estimate"]
    assert run(task, tools, default_registry(), ledger).looks == out.looks   # A finished run does not run again.


# ---------------------------------------------------------------- against a fixed budget

def test_against_fixed_budget_the_agent_is_cheaper_and_still_covers_the_truth():
    covered, saved = 0, 0
    for seed in range(10):
        out, _, _, _ = _go(FAST, seed=seed)
        fixed, _, _, _ = _go(FAST, "auto", (), seed=seed)       # No early stop: the whole budget.
        a = out.assessments["x"]
        covered += abs(a["estimate"]["a"] - 2 / 3) <= 2 * a["uncertainty"]["a"]
        assert out.spent <= fixed.spent
        saved += out.spent < fixed.spent
    assert saved >= 8 and covered >= 8
