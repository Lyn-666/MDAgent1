"""Step 4: the next action after each round, and the permission check before it."""

import json
from dataclasses import asdict

import pytest

import mdagent.agent as agent
from mdagent.agent import Decision, Goal, LoopSettings, RunState
from mdagent.diagnose import ResidueReport
from mdagent.messages import read
from mdagent.policy import plain_md_next
from mdagent.simulate import MDSettings


def _state(tmp_path, permissions, ns=(1.0, 5.0), max_ns=20.0, replicas=3):
    s = RunState(pdb="x", target_description="t", target=[("A", 111)], n_replicas=replicas, base_seed=1,
                 md=MDSettings(), loop=LoopSettings(max_ns=max_ns, extend_ns=5.0), rounds_ns=list(ns),
                 goal=asdict(Goal(permissions=permissions)))
    s.save(tmp_path)
    return s


def _report(verdict, transitions=(6, 6, 6), differ=False):
    return ResidueReport(chain="A", resseq=111, resname="VAL", transitions=list(transitions), occupancies=[],
                         dominant=["t"] * len(transitions), verdict=verdict, replicas_differ=differ,
                         error_from_replicas=differ)


def test_rule1_stop_when_done_unless_stopping_early_is_not_allowed(tmp_path):
    done = [Decision("VAL111", "done", "ok")]
    state = _state(tmp_path, ["stop_early"])
    assert plain_md_next(tmp_path, state, [_report("ok")], done, 0.0) == 0.0
    assert state.notes["actions"][-1]["action"] == "stop"
    state = _state(tmp_path, [])
    assert plain_md_next(tmp_path, state, [_report("ok")], done, 0.0) == 5.0      # Goes on to the budget.
    assert "没有允许提前停" in read(tmp_path)[-1]["zh"]


def test_rule3_needs_real_disagreement_and_the_first_half_of_the_budget(tmp_path):
    ext = [Decision("VAL111", "extend", "imprecise")]
    noisy = _report("imprecise")
    noisy.error_from_replicas = True              # Spread larger than the inner error, but no clear disagreement.
    state = _state(tmp_path, ["add_replicas"])
    plain_md_next(tmp_path, state, [noisy], ext, 5.0)
    assert state.n_replicas == 3
    late = _state(tmp_path, ["add_replicas"], ns=(1.0, 5.0, 5.0))   # 11 of 20 ns: too late to catch up.
    plain_md_next(tmp_path, late, [_report("imprecise", differ=True)], ext, 5.0)
    assert late.n_replicas == 3


def test_rule3_add_replicas_when_the_error_comes_from_their_spread(tmp_path):
    ext = [Decision("VAL111", "extend", "imprecise")]
    state = _state(tmp_path, ["add_replicas"])
    assert plain_md_next(tmp_path, state, [_report("imprecise", differ=True)], ext, 5.0) == 5.0
    assert state.n_replicas == 5 and state.notes["actions"][-1] == {**state.notes["actions"][-1],
                                                                   "action": "add_replicas", "executed": True}
    state = _state(tmp_path, [], replicas=3)                                         # No permission.
    plain_md_next(tmp_path, state, [_report("imprecise", differ=True)], ext, 5.0)
    assert state.n_replicas == 3 and not state.notes["actions"][-1]["executed"]
    assert read(tmp_path)[-1]["kind"] == "suggestion"
    state = _state(tmp_path, ["add_replicas"], replicas=10)                          # At the limit.
    plain_md_next(tmp_path, state, [_report("imprecise", differ=True)], ext, 5.0)
    assert state.n_replicas == 10


def test_rule4_extend_beyond_the_budget_only_with_permission_and_up_to_the_limit(tmp_path):
    def used_up():
        return [Decision("VAL111", "enhanced_sampling", "budget", cause="budget_exhausted")]
    state = _state(tmp_path, ["extend"], ns=(10.0, 10.0), max_ns=20.0)
    d = used_up()
    assert plain_md_next(tmp_path, state, [_report("imprecise")], d, 0.0) == 5.0
    assert state.loop.max_ns == 25.0 and d[0].action == "extend"
    state = _state(tmp_path, ["extend"], ns=(20.0, 20.0), max_ns=40.0)               # At 2 x the first budget.
    state.notes["first_max_ns"] = 20.0
    assert plain_md_next(tmp_path, state, [_report("imprecise")], used_up(), 0.0) == 0.0
    state = _state(tmp_path, [], ns=(10.0, 10.0), max_ns=20.0)                       # No permission.
    assert plain_md_next(tmp_path, state, [_report("imprecise")], used_up(), 0.0) == 0.0
    assert state.notes["actions"][-1]["action"] == "extend_budget" and not state.notes["actions"][-1]["executed"]
    # A stuck item is not a reason to run longer: switching or suggesting is the answer.
    stuck = [Decision("VAL111", "enhanced_sampling", "no flip", cause="stalled")]
    state = _state(tmp_path, ["extend"], ns=(10.0, 10.0), max_ns=20.0)
    assert plain_md_next(tmp_path, state, [_report("no_flip", (0, 0, 0))], stuck, 0.0) == 0.0


def test_rule5_suggest_reseeding_when_replicas_sit_in_different_states(tmp_path):
    split = ResidueReport(chain="A", resseq=111, resname="VAL", transitions=[0, 0, 0], occupancies=[],
                          dominant=["t", "g+", "t"], verdict="replicas_disagree")
    state = _state(tmp_path, list(agent.PERMISSIONS))
    plain_md_next(tmp_path, state, [split], [Decision("VAL111", "extend", "split")], 5.0)
    assert state.notes["actions"][-1]["action"] == "suggest_reseed" and not state.notes["actions"][-1]["executed"]
    assert "自适应采样" in read(tmp_path)[-1]["zh"]


def test_added_replicas_catch_up_before_the_next_round(tmp_path, monkeypatch):
    calls = []

    def fake_segment(files, run_dir, i, part, seed, ns, md, log, start_frame=None, full_interval_ps=None):
        calls.append((i, part))
        (run_dir / f"replica_{i}_part{part:03d}.state.xml").write_text("x")
    monkeypatch.setattr(agent, "run_segment", fake_segment)
    rounds = iter([2, 0])  # After round 1 the policy adds 2 replicas (3 -> 5); after round 2 it stops.

    def fake_decide(run_dir, state):
        if next(rounds) == 2:
            state.n_replicas = 5
            return [], [Decision("VAL111", "extend", "x")], 1.0
        return [], [Decision("VAL111", "done", "ok")], 0.0
    monkeypatch.setattr(agent, "_diagnose_and_decide", fake_decide)
    monkeypatch.setattr(agent, "_record", lambda *a, **k: "")
    monkeypatch.setattr(agent, "format_report", lambda *a, **k: "")
    state = _state(tmp_path, list(agent.PERMISSIONS), ns=(), replicas=3)
    state.loop.first_ns = 1.0
    agent.run_loop(tmp_path, state, files=object(), log=lambda line: None)
    assert calls[:3] == [(0, 0), (1, 0), (2, 0)]
    # Round 2: replicas 3 and 4 first run round 1, then everyone runs round 2.
    assert calls[3:] == [(0, 1), (1, 1), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]


def test_rule6_stop_early_when_the_limit_is_too_short_to_reach_the_goal(tmp_path):
    from mdagent.policy import projected_ns

    slow = _report("rare_flips", transitions=(1, 0, 2))            # 1 change per replica in 6 ns: needs ~24 ns.
    slow.standard_error = {"g+": 0.02, "t": 0.03, "g-": 0.01}
    assert 20 < projected_ns(slow, 6.0, 0.05, 4) < 30

    ext = [Decision("VAL111", "extend", "rare flips")]
    state = _state(tmp_path, ["stop_early"], ns=(1.0, 5.0), max_ns=10.0)   # Limit 10 ns, needs ~24 ns.
    assert plain_md_next(tmp_path, state, [slow], ext, 4.0) == 0.0
    assert ext[0].cause == "not_reachable" and ext[0].action == "enhanced_sampling"
    assert state.notes["projected_ns"]["VAL111"] > 15
    assert "提前停下" in read(tmp_path)[-1]["zh"]

    # The same with a limit the estimate fits into: go on.
    ext = [Decision("VAL111", "extend", "rare flips")]
    state = _state(tmp_path, ["stop_early"], ns=(1.0, 5.0), max_ns=20.0)
    assert plain_md_next(tmp_path, state, [slow], ext, 5.0) == 5.0 and ext[0].action == "extend"

    # Guided (no stop_early): only say so, and go on to the limit.
    ext = [Decision("VAL111", "extend", "rare flips")]
    state = _state(tmp_path, [], ns=(1.0, 5.0), max_ns=10.0)
    assert plain_md_next(tmp_path, state, [slow], ext, 4.0) == 4.0 and ext[0].action == "extend"
    assert "按你的设置继续" in read(tmp_path)[-1]["zh"]

    # Too early to judge (one round): go on.
    ext = [Decision("VAL111", "extend", "rare flips")]
    state = _state(tmp_path, ["stop_early"], ns=(1.0,), max_ns=10.0)
    assert plain_md_next(tmp_path, state, [slow], ext, 5.0) == 5.0


def test_limits_are_not_passed_without_a_yes_and_the_message_says_how(tmp_path):
    # Budget used up, more data needed, no permission to extend: suggest a length, do not extend.
    ext = [Decision("VAL111", "extend", "imprecise")]
    state = _state(tmp_path, ["stop_early"], ns=(1.0, 5.0, 4.0), max_ns=10.0)
    rep = _report("imprecise", transitions=(8, 9, 10))
    rep.standard_error = {"g+": 0.07, "t": 0.03, "g-": 0.01}
    assert plain_md_next(tmp_path, state, [rep], ext, 0.0) == 0.0 and state.loop.max_ns == 10.0
    assert "加长到" in read(tmp_path)[-1]["zh"]
    # Replicas disagree, no permission to add: say how to confirm.
    ext = [Decision("VAL111", "extend", "imprecise")]
    state = _state(tmp_path, ["stop_early"])
    plain_md_next(tmp_path, state, [_report("imprecise", differ=True)], ext, 5.0)
    assert state.n_replicas == 3 and "加 2 条重复" in read(tmp_path)[-1]["zh"]


def test_no_change_after_the_patience_time_closes_the_item_when_stopping_early_is_allowed():
    from mdagent.agent import decide

    stuck = _report("no_flip", transitions=(0, 0, 0))
    loop = LoopSettings(max_ns=20.0, no_flip_patience_ns=5.0)
    decisions, next_ns = decide([stuck], 6.0, loop, may_switch=False, may_stop=True)
    assert next_ns == 0.0 and decisions[0].cause == "stalled"
    decisions, next_ns = decide([stuck], 6.0, loop, may_switch=False, may_stop=False)   # Guided: goes on.
    assert next_ns > 0 and decisions[0].stalled
