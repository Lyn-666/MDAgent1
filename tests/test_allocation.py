"""The agent's allocation rule (allocation.py) and how the live run loop uses it."""

import json
from dataclasses import asdict

import mdtraj as md
import numpy as np
import pytest

from mdagent.allocation import CONTINUE, INSUFFICIENT, STOP, ItemState, Limits, next_step

LIM = Limits(max_replicas=8, max_ns=100.0, budget_ns=800.0, chunk_ns=5.0, patience_ns=20.0)


def _item(label="x", se=0.01, w=0.01, b=0.01, trans=(20, 20, 20, 20), ok=False):
    return ItemState(label, {"a": se, "b": se}, w, b, list(trans), ok)


def test_the_first_look_only_plans_even_when_everything_looks_precise():
    s = next_step([_item(ok=True)], 4, 5.0, look=1, tests_before=0, z=2.39, tolerance=0.05, limits=LIM)
    assert s.action == CONTINUE and s.planning
    s2 = next_step([_item(ok=True)], 4, 10.0, look=2, tests_before=0, z=2.39, tolerance=0.05, limits=LIM)
    assert s2.action == STOP and s2.kind == "converged"


def test_it_says_insufficient_evidence_instead_of_running_on():
    stuck = _item(trans=(0, 0, 0, 0))
    s = next_step([stuck], 4, 40.0, look=2, tests_before=0, z=2.39, tolerance=0.05, limits=LIM)
    assert s.action == INSUFFICIENT and s.kind == "stuck"   # 160 ns unchanged: ~1100 ns needed at best
    s = next_step([_item()], 4, 20.0, look=4, tests_before=2, z=2.39, tolerance=0.05, limits=LIM)
    assert s.action == INSUFFICIENT and s.kind == "looks_used"
    s = next_step([_item()], 8, 100.0, look=2, tests_before=0, z=2.39, tolerance=0.05, limits=LIM)
    assert s.action == INSUFFICIENT and s.kind == "budget_used"


def test_it_runs_longer_when_the_error_is_inside_the_replicas():
    s = next_step([_item(se=0.03, w=0.03, b=0.03)], 4, 10.0, look=2, tests_before=0, z=2.39, tolerance=0.05,
                  limits=LIM)
    assert s.action == CONTINUE and s.replicas == 4 and s.ns > 10.0 and s.kind == "projected"


def test_it_adds_replicas_when_they_really_disagree():
    s = next_step([_item(se=0.025, w=0.01, b=0.025)], 4, 10.0, look=2, tests_before=0, z=2.39, tolerance=0.05,
                  limits=LIM)
    assert s.action == CONTINUE and s.replicas > 4


def test_it_stays_inside_the_budget():
    s = next_step([_item(se=0.04, w=0.04, b=0.04)], 4, 10.0, look=2, tests_before=0, z=2.39, tolerance=0.05,
                  limits=Limits(max_replicas=8, max_ns=100.0, budget_ns=120.0))
    assert s.replicas * s.ns <= 120.0 + 1e-9


# --- the live loop -------------------------------------------------------------------------------

def _backbone_run(tmp_path, n_replicas=3, frames=20):
    """A run folder with backbone trajectories (residues 50-54 of 181L, jittered) and a projection goal."""
    from mdagent.agent import Goal, LoopSettings, RunState
    from mdagent.simulate import MDSettings

    t = md.load("examples/181L.pdb")
    part = t.atom_slice([a.index for a in t.topology.atoms
                         if a.residue.chain.chain_id == "A" and 50 <= a.residue.resSeq <= 54])
    part.save_pdb(str(tmp_path / "protein.pdb"))
    rng = np.random.default_rng(0)
    for i in range(n_replicas):
        xyz = part.xyz[0] + rng.normal(0, 0.002, (frames, part.n_atoms, 3)).astype(np.float32)
        md.Trajectory(xyz, part.topology).save_dcd(str(tmp_path / f"replica_{i}_part000.dcd"))
    state = RunState(pdb="x", target_description="backbone", target=[("A", n) for n in range(50, 55)],
                     n_replicas=n_replicas, base_seed=0, md=MDSettings(save_interval_ps=5.0),
                     loop=LoopSettings(first_ns=0.1, extend_ns=0.1, max_ns=1.0, no_flip_patience_ns=0.5),
                     rounds_ns=[0.1], goal=asdict(Goal(quantity="backbone_basins", allocation="projection")),
                     backbone=["rama:A:52", "turn:A:52"], notes={"replicas_at_start": n_replicas})
    state.save(tmp_path)
    return state


def test_the_live_loop_asks_the_allocation_rule_once_per_look(tmp_path):
    from mdagent.agent import _diagnose_and_decide

    state = _backbone_run(tmp_path)
    reports, decisions, next_ns = _diagnose_and_decide(tmp_path, state)
    assert {r.label for r in reports} == {"rama:A:52", "turn:A:52"}
    assert all(r.quantity == "backbone" and r.extra and "rates" in r.extra for r in reports)
    assert next_ns > 0                                         # the first look plans: it never stops
    assert state.notes["allocation"] == {"looks": 1, "tests": 0, "counting": 2}
    act = state.notes["actions"][-1]
    assert act["phase"] == "plain_md" and "planning only" in act["reason"]
    # Resumed at the same point: the same choice, no second look counted, no second message.
    again = _diagnose_and_decide(tmp_path, state)[2]
    assert again == next_ns and state.notes["allocation"]["looks"] == 1
    msgs = [json.loads(l) for l in (tmp_path / "messages.jsonl").read_text().splitlines()]
    assert sum("第 1 次查看" in m["zh"] for m in msgs) == 1


def test_projection_runs_are_calibrated_for_their_stop_looks(tmp_path):
    from mdagent.allocation import z_bonferroni

    state = _backbone_run(tmp_path)
    p = state.precision()
    assert p.criterion == "confidence" and p.looks == 3 and p.z == pytest.approx(z_bonferroni(0.05, 3))
    assert state.budget_ns() == 10 * 1.0                       # the user's limits: at most 10 replicas x 1 ns
    state.goal["permissions"] = ["stop_early"]
    assert state.budget_ns() == 3 * 1.0                        # no adding replicas: the 3 it starts with
    assert state.max_ns_per_replica() == 1.0


def test_a_chat_plan_for_backbone_basins_becomes_a_valid_run():
    from mdagent.cli import build_parser
    from mdagent.llm import RunPlan, to_argv

    plan = RunPlan(target_kind="backbone", backbone="rama:A:8-10;turn:A:52", allocation="projection",
                   allow="extend,add_replicas,stop_early", budget_ns=400, questions=[], summary="x")
    argv = to_argv(plan, "structures/1UBQ.pdb", "runs/ub")
    assert argv[argv.index("--backbone") + 1] == "rama:A:8-10;turn:A:52"
    assert argv[argv.index("--allocation") + 1] == "projection" and "--budget-ns" in argv
    args = build_parser().parse_args(argv)
    assert args.quantity == "backbone" and args.allocation == "projection"


def test_the_report_explains_the_decisions_and_judges_each_observable(tmp_path):
    from mdagent.report import write

    (tmp_path / "run.json").write_text(json.dumps({
        "pdb": "x", "target_description": "backbone", "n_replicas": 4, "rounds_ns": [5.0, 25.0],
        "md": {"temperature_k": 300.0, "equilibration_ps": 600.0}, "loop": {"first_ns": 5.0, "max_ns": 50.0},
        "backbone": ["rama:A:9"], "setup": {},
        "goal": {"quantity": "backbone_basins", "precision": 0.05, "allocation": "projection", "stop_looks": 3,
                 "budget_ns": 400, "max_replicas": 8, "permissions": ["extend", "add_replicas", "stop_early"]},
        "notes": {"replicas_at_start": 4,
                  "actions": [{"time": "t1", "phase": "plain_md", "action": "extend", "executed": True,
                               "reason": "Look 1 (planning only) at 4 x 5 ns: projected. Next: 4 x 30 ns.",
                               "ns_per_replica": 5.0}],
                  "allocation_decision": {"decision": "STOP", "reason": "every item is precise enough", "look": 2},
                  "plain_md_outcome": {"reason": "target_reached", "ns_per_replica": 30.0, "budget_ns": 50.0}}}))
    (tmp_path / "status.json").write_text(json.dumps({"residues": [{
        "chain": "A", "resseq": 9, "resname": "", "name": "rama:A:9", "verdict": "ok", "quantity": "backbone",
        "transitions": [40, 38, 41, 39], "populations": {"alphaR": 0.3, "beta": 0.65, "alphaL": 0.05},
        "standard_error": {"alphaR": 0.012, "beta": 0.013, "alphaL": 0.005},
        "extra": {"dg_kt": -0.77, "dg_se": 0.06, "split_js": 0.01, "angles": ["phi of residue 9", "psi of residue 9"],
                  "rates": {"alphaR->beta": {"k_per_ns": 1.2, "n": 30, "rel_se": 0.18},
                            "beta->alphaR": {"k_per_ns": 0.6, "n": 29, "rel_se": 0.19}}}}]}))
    zh, en = (p.read_text() for p in write(tmp_path))
    assert "Backbone basins: every observable judged on its own" in en
    assert "| Basin populations | alphaR 0.300 ± 0.029" in en and "| converged |" in en
    assert "not converged" in en                     # the rates are too uncertain at 30 transitions
    assert "landscape_rama_A_9.svg" in en
    assert "The agent's decisions" in en and "**Stop:** STOP" in en and "planning only" in en
    assert "of the budget" in en and "against the budget, not against a fixed plan" in en
    assert "主链构象区：每个量单独判断" in zh and "agent 的决策" in zh


@pytest.mark.slow
def test_a_live_backbone_run_decides_and_says_why(tmp_path):
    """Tiny peptide on CPU: the live loop with projection allocation ends with STOP or INSUFFICIENT_EVIDENCE."""
    from mdagent.agent import Goal, LoopSettings, RunState, run_all
    from mdagent.build import SetupOptions, build_system
    from mdagent.report import write
    from mdagent.simulate import MDSettings, SystemFiles, write_inputs

    built = build_system("tests/data/fragment_108-114.pdb", SetupOptions(forcefield="amber14", padding_nm=0.8))
    write_inputs(built, tmp_path)
    state = RunState(pdb="x", target_description="backbone", target=[("A", n) for n in range(109, 114)],
                     n_replicas=2, base_seed=1, md=MDSettings(equilibration_ps=1.0, save_interval_ps=0.5, platform="CPU"),
                     loop=LoopSettings(first_ns=0.002, extend_ns=0.002, max_ns=0.006, no_flip_patience_ns=0.004),
                     goal=asdict(Goal(quantity="backbone_basins", allocation="projection", precision=0.15)),
                     backbone=["rama:A:110"], notes={"replicas_at_start": 2})
    state.save(tmp_path)
    run_all(tmp_path, state, SystemFiles(built.topology, built.positions, built.system), log=lambda line: None)
    saved = RunState.load(tmp_path)
    assert saved.notes["allocation"]["looks"] >= 2
    assert saved.notes["plain_md_outcome"]["decision"] in ("STOP", "INSUFFICIENT_EVIDENCE")
    assert saved.ns_done <= saved.max_ns_per_replica() + 1e-9
    paths = write(tmp_path)
    assert (tmp_path / "landscape_rama_A_110.svg").exists()
    assert "The agent's decisions" in paths[1].read_text()


def test_an_item_plain_md_cannot_answer_does_not_stop_the_others():
    """A slow turn that never flips is set aside; the residues that do flip go on to their precision."""
    slow = _item("turn:A:52", trans=(0, 0, 0, 0))
    fast = _item("rama:A:9", se=0.03, w=0.03, b=0.03)
    s = next_step([slow, fast], 4, 40.0, look=2, tests_before=0, z=2.39, tolerance=0.05, limits=LIM)
    assert s.action == CONTINUE and s.set_aside == {"turn:A:52": "stuck"} and s.ns > 40.0
    done = _item("rama:A:9", ok=True)
    s = next_step([slow, done], 4, 60.0, look=3, tests_before=1, z=2.39, tolerance=0.05, limits=LIM)
    assert s.action == STOP and "turn:A:52" in s.set_aside and "Left out" in s.reason
    s = next_step([slow], 4, 40.0, look=2, tests_before=0, z=2.39, tolerance=0.05, limits=LIM)
    assert s.action == INSUFFICIENT                     # nothing answerable left: the whole run says so


def test_a_look_that_could_not_stop_does_not_use_up_the_looks():
    """Too few state changes: STOP is impossible there, so the look spends none of alpha."""
    few = _item(trans=(1, 2, 1, 0))
    s = next_step([few], 4, 2.0, look=2, tests_before=0, z=2.39, tolerance=0.05, limits=LIM)
    assert s.action == CONTINUE and not s.counted and s.ns > 2.0
    # Even after many such looks the run goes on; only looks that could stop are counted.
    s = next_step([few], 4, 8.0, look=6, tests_before=0, z=2.39, tolerance=0.05, limits=LIM)
    assert s.action == CONTINUE and not s.counted
    ready = _item(se=0.03, w=0.03, b=0.03)
    s = next_step([ready], 4, 10.0, look=7, tests_before=0, z=2.39, tolerance=0.05, limits=LIM)
    assert s.counted


def test_an_older_run_gets_its_looks_counted_again(tmp_path):
    """A run started before the fix counted looks that could not stop; resuming it counts them again."""
    from mdagent.policy import recount_tests

    actions = [
        {"phase": "plain_md", "reason": "Look 1 (planning only) at 3 × 1 ns: too few state changes to trust the "
                                        "errors yet; running longer. Next: 3 × 2 ns."},
        {"phase": "plain_md", "reason": "Look 2 at 3 × 2 ns: too few state changes to trust the errors yet; "
                                        "running longer. Next: 3 × 4 ns."},
        {"phase": "plain_md", "reason": "Look 3 at 3 × 4 ns: too few state changes to trust the errors yet; "
                                        "running longer. Next: 3 × 16 ns."},
        {"phase": "plain_md", "reason": "Look 4 at 3 × 16 ns: projected: 3 replicas of 30 ns reach the precision."},
    ]
    assert recount_tests(actions[:3]) == 0
    assert recount_tests(actions) == 1

    from mdagent.agent import _diagnose_and_decide

    state = _backbone_run(tmp_path)
    state.notes["allocation"] = {"looks": 3, "tests": 2}          # as the old code left it
    state.notes["actions"] = actions[:3]
    state.save(tmp_path)
    _diagnose_and_decide(tmp_path, state)
    assert state.notes["allocation"]["counting"] == 2
    assert state.notes["allocation"]["tests"] == 0               # the jittered frames cannot stop either


def test_an_unchanged_item_is_given_up_only_when_even_the_fastest_rate_misses_the_budget():
    """Zero changes in A ns: at best one change per A/3 ns. Give up only if the ~20 changes the precision
    needs would then still take more than the budget."""
    from mdagent.allocation import stuck_projection

    rate, need, data = stuck_projection(3, 16.0, 2.39, 0.05)
    assert rate == pytest.approx(3 / 48) and 19 < need < 22 and data == pytest.approx(need * 16)
    still = _item("rama:A:9", trans=(0, 0, 0))
    big = Limits(max_replicas=8, max_ns=100.0, budget_ns=800.0, patience_ns=10.0)
    s = next_step([still], 3, 16.0, look=4, tests_before=0, z=2.39, tolerance=0.05, limits=big)
    assert s.action == CONTINUE and not s.set_aside            # 48 ns: ~330 ns needed at best, budget 800
    s = next_step([still], 3, 64.0, look=6, tests_before=0, z=2.39, tolerance=0.05, limits=big)
    assert s.action == INSUFFICIENT and s.set_aside == {"rama:A:9": "stuck"}   # 192 ns: ~1300 ns needed
    assert "one change per 64 ns" in s.reason and "more than the budget of 800 ns" in s.reason


def test_a_finished_run_decides_again_under_a_newer_rule(tmp_path):
    from mdagent.agent import _apply_policy

    state = _backbone_run(tmp_path)
    old_key = {"ns": state.ns_done, "max_ns": state.loop.max_ns, "n_replicas": state.n_replicas,
               "permissions": sorted(state.the_goal().permissions)}
    state.notes["policy_round"] = {"key": old_key, "next_ns": 0.0, "decisions": {}}   # ended under the old rule
    from mdagent.agent import _diagnose_and_decide

    _, _, next_ns = _diagnose_and_decide(tmp_path, state)
    assert next_ns > 0 and state.notes["policy_round"]["key"]["rule"] >= 3
