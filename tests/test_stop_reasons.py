"""Why the agent stops: goal reached, sampling stalled, budget used up, or the run failed.

The simulation is replaced by a stand-in that only records progress, and the analysis
returns prepared results, so each scenario checks the decision logic itself.
"""

import json
from dataclasses import asdict

import pytest

import mdagent.agent as agent
import mdagent.enhanced as enhanced
import mdagent.reweight as reweight
from mdagent.agent import Decision, Goal, LoopSettings, RunState, run_all
from mdagent.messages import read
from mdagent.simulate import MDSettings


def _state(tmp_path, permissions):
    state = RunState(pdb="x", target_description="t", target=[("A", 111)], n_replicas=2, base_seed=1,
                     md=MDSettings(), loop=LoopSettings(), rounds_ns=[20.0],
                     goal=asdict(Goal(precision=0.05, permissions=permissions)),
                     enhanced={"enabled": True, "settings": {"length_ns": 6.0, "chunk_ns": 1.0, "n_replicas": 2},
                               "stop_rule": {"min_ns": 2.0}})
    state.save(tmp_path)
    return state


def _result(ns, se, transitions):
    pops = {"g+": 0.2, "t": 0.7, "g-": 0.1}
    return {"ns_per_replica": ns, "n_replicas": 2, "effective_samples": [500.0, 500.0], "weights_degenerate": False,
            "method": "test", "residues": [{
                "residue": "VAL111", "populations": pops, "standard_error": {k: se for k in pops},
                "max_standard_error": se, "transitions_per_replica": list(transitions), "replicas_disagree": False}]}


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """Returns run(state, se_at, transitions) -> (ns run per replica, outcome, messages)."""
    (tmp_path / "enhanced").mkdir()
    flagged = [Decision("VAL111", "enhanced_sampling", "no flip", ("A", 111), cause="stalled")]
    monkeypatch.setattr(agent, "run_loop", lambda *a, **k: flagged)

    def fake_replica(files, run_dir, r, start, residues, s, md, seed, log, until_ns):
        (run_dir / "enhanced" / f"replica_{r}.json").write_text(json.dumps({"ns_done": until_ns}))
    monkeypatch.setattr(enhanced, "run_replica", fake_replica)

    def run(state, se_at, transitions=(40, 40)):
        monkeypatch.setattr(reweight, "analyze",
                            lambda run_dir, labels, n, s, t, rule: _result(enhanced.ns_done(run_dir, 0),
                                                                           se_at(enhanced.ns_done(run_dir, 0)),
                                                                           transitions))
        run_all(tmp_path, state, files=object(), log=lambda line: None)
        saved = RunState.load(tmp_path)
        return enhanced.ns_done(tmp_path, 0), saved.enhanced["outcome"], read(tmp_path)
    return run


def test_goal_reached_stops_early_when_allowed(tmp_path, fake_run):
    ns, outcome, msgs = fake_run(_state(tmp_path, ["switch_method", "stop_early"]), se_at=lambda ns: 0.08 if ns < 3 else 0.03)
    assert ns == 3.0 and outcome["reason"] == "target_reached" and outcome["residues"] == {"VAL111": "reached"}
    assert msgs[-1]["kind"] == "done" and "达到目标" in msgs[-1]["zh"]


def test_all_replicas_trapped_is_stalled_not_converged(tmp_path, fake_run):
    # Tiny errors and identical replicas, but no transitions: never "reached".
    ns, outcome, msgs = fake_run(_state(tmp_path, ["switch_method", "stop_early"]), se_at=lambda ns: 0.0, transitions=(0, 0))
    # Stuck at the checks after 2, 3 and 4 ns (min_ns is 2): it stops early instead of using all 6 ns.
    assert ns == 4.0 and outcome["reason"] == "stalled"
    actions = RunState.load(tmp_path).notes["actions"]
    assert actions[-1]["action"] == "stop" and "3 checks" in actions[-1]["reason"]
    assert any("推不动" in m["zh"] for m in msgs) and msgs[-1]["kind"] == "suggestion"


def test_budget_used_up_before_the_precision(tmp_path, fake_run):
    ns, outcome, msgs = fake_run(_state(tmp_path, ["switch_method", "stop_early"]), se_at=lambda ns: 0.08)
    assert ns == 6.0 and outcome["reason"] == "budget_exhausted" and outcome["residues"] == {"VAL111": "not_yet"}
    assert "预算用完" in msgs[-1]["zh"]


def test_without_permission_it_runs_the_full_budget_and_says_when_the_goal_is_met(tmp_path, fake_run):
    ns, outcome, msgs = fake_run(_state(tmp_path, ["switch_method"]), se_at=lambda ns: 0.08 if ns < 3 else 0.03)
    assert ns == 6.0 and outcome["reason"] == "target_reached"
    assert [m["kind"] for m in msgs] == ["info", "done"]          # "reached, going on as you set it", then done.


def test_a_failure_is_recorded_as_a_stop_reason(tmp_path):
    from mdagent.agent import record_failure

    _state(tmp_path, [])
    record_failure(tmp_path, RuntimeError("CUDA error"))
    assert RunState.load(tmp_path).notes["outcome"]["reason"] == "failed"
    assert "CUDA error" in read(tmp_path)[-1]["en"]


def test_plain_md_stop_reasons(tmp_path):
    from mdagent.agent import _post_plain_md_messages

    for decisions, reason in [
        ([Decision("VAL111", "done", "ok")], "target_reached"),
        ([Decision("VAL111", "enhanced_sampling", "no flip", cause="stalled"),
          Decision("LEU84", "enhanced_sampling", "budget", cause="budget_exhausted")], "stalled"),
        ([Decision("LEU84", "enhanced_sampling", "budget", cause="budget_exhausted")], "budget_exhausted"),
    ]:
        state = _state(tmp_path, list(agent.PERMISSIONS))
        state.notes = {}
        _post_plain_md_messages(tmp_path, state, decisions, 0.0)
        assert state.notes["plain_md_outcome"]["reason"] == reason


def test_enhanced_adds_a_replica_when_the_spread_dominates(tmp_path, fake_run, monkeypatch):
    base = _result

    def spread_result(ns, se, transitions):
        res = base(ns, se, transitions)
        res["residues"][0]["error_from_replicas"] = True
        return res
    monkeypatch.setattr(__import__(__name__), "_result", spread_result)
    state = _state(tmp_path, ["switch_method", "stop_early", "add_replicas"])
    ns, outcome, msgs = fake_run(state, se_at=lambda ns: 0.08 if ns < 3 else 0.03)
    saved = RunState.load(tmp_path)
    assert saved.enhanced["settings"]["n_replicas"] == 4                  # One more at the checks after 1 and 2 ns.
    assert any(a["action"] == "add_replicas" and a["executed"] for a in saved.notes["actions"])
    assert json.loads((tmp_path / "enhanced" / "replica_3.json").read_text())["ns_done"] == 3.0  # Caught up.


def test_enhanced_runs_longer_than_its_budget_only_with_permission(tmp_path, fake_run):
    ns, outcome, _ = fake_run(_state(tmp_path, ["switch_method", "stop_early", "extend"]),
                              se_at=lambda ns: 0.08 if ns < 8 else 0.03)
    assert ns == 8.0 and outcome["reason"] == "target_reached"               # 6 ns budget, extended.
    saved = RunState.load(tmp_path)
    assert any(a["action"] == "extend_budget" and a["executed"] for a in saved.notes["actions"])
