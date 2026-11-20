"""The budget comparison: does a strategy reach the asked precision for less computing?

The pool here is a Markov model with known populations, so every answer can be checked
against the truth instead of against another estimate.
"""

import numpy as np
import pytest

from mdagent.budget import (Plan, Pool, check_against, compare_strategies, format_table, markov_pool,
                            reference, replay, write)
from mdagent.cost import CostModel
from mdagent.uncertainty import Precision

FAST = dict(rate_per_ns=5.0, n_replicas=16, ns=60.0)
PLAN = Plan(replicas=3, first_ns=2.0, extend_ns=2.0, max_ns=20.0, extend_factor=2.0,
            fixed_replicas=5, fixed_ns=20.0)


def _precision(criterion="standard_error", epsilon=0.05):
    return Precision(epsilon=epsilon, criterion=criterion, looks=PLAN.looks())


def test_the_model_pool_has_the_populations_it_was_given():
    pool = markov_pool([0.3, 0.7], **FAST, seed=3)
    all_frames = np.concatenate(pool.series["model"])
    assert abs((all_frames == 0).mean() - 0.3) < 0.02
    assert pool.ns_per_replica == 60.0 and pool.n_replicas == 16
    assert pool.truth == {"model": {"s0": 0.3, "s1": 0.7}}


def test_a_strategy_pays_for_every_replica_it_keeps():
    pool = markov_pool([0.3, 0.7], **FAST, seed=1)
    t = replay(pool, "fixed", PLAN, _precision(), CostModel(equilibration_ns=0.6))
    assert t.cost["n_replicas"] == 5 and t.ns_per_replica == 20.0
    assert t.cost["aggregate_ns"] == 100.0
    assert t.cost["compute_ns"] == 103.0          # Five equilibrations of 0.6 ns.
    assert t.steps[-1].action == "stop"


def test_a_replica_added_late_still_has_to_catch_up():
    """Its production time counts in full, as run_loop really makes it catch up."""
    pool = markov_pool([0.3, 0.7], rate_per_ns=5.0, n_replicas=16, ns=60.0, seed=7)
    t = replay(pool, "add_replicas", PLAN, _precision(epsilon=0.01), CostModel(equilibration_ns=0.0))
    assert t.cost["n_replicas"] > PLAN.replicas
    assert t.cost["aggregate_ns"] == pytest.approx(t.cost["n_replicas"] * t.ns_per_replica)


def test_stopping_at_the_right_time_costs_less_than_the_fixed_plan():
    pool = markov_pool([0.3, 0.7], **FAST, seed=0)
    fixed = replay(pool, "fixed", PLAN, _precision(), CostModel())
    early = replay(pool, "early_stop", PLAN, _precision(), CostModel())
    assert early.cost["compute_ns"] < fixed.cost["compute_ns"]
    assert early.stop_reason == "target_reached" and early.met


def test_the_agent_stops_on_an_item_plain_md_cannot_answer():
    """A frozen item: plain MD can run forever without learning anything, so the agent stops."""
    pool = markov_pool([0.3, 0.7], **FAST, seed=1)
    frozen = np.zeros_like(pool.series["model"][0])
    pool = Pool(["frozen"], {"frozen": ["s0", "s1"]}, {"frozen": [frozen.copy() for _ in range(16)]},
                pool.ns_per_frame, source="frozen item")
    plan = Plan(**{**PLAN.__dict__, "no_flip_patience_ns": 6.0})
    agent = replay(pool, "mdagent", plan, _precision(), CostModel())
    naive = replay(pool, "early_stop", plan, _precision(), CostModel())
    assert agent.stop_reason == "stalled" and agent.unresolved == ["frozen"]
    assert agent.ns_per_replica <= 8.0
    assert naive.cost["compute_ns"] > agent.cost["compute_ns"]   # The naive one spends the whole budget.


def test_the_check_catches_an_answer_that_is_outside_the_tolerance():
    pool = markov_pool([0.3, 0.7], **FAST, seed=1)
    t = replay(pool, "early_stop", PLAN, _precision(), CostModel())
    ref, used, independent = reference(pool, t.replicas, PLAN, _precision())
    assert independent and used == []            # A model pool is checked against the known truth.
    check_against(t, ref, 0.05)
    assert t.accurate is not None
    # A made-up answer that is 0.2 away must fail, however small its error claims to be.
    t.results["model"]["populations"] = {"s0": 0.1, "s1": 0.9}
    t.results["model"]["standard_error"] = {"s0": 0.001, "s1": 0.001}
    check_against(t, ref, 0.05)
    assert t.accurate is False and t.equivalent is False
    assert t.check["worst_difference"] == pytest.approx(0.2)


def test_held_out_replicas_are_used_when_the_truth_is_not_known():
    pool = markov_pool([0.3, 0.7], **FAST, seed=1)
    pool.truth = None
    ref, used, independent = reference(pool, [0, 1, 2], PLAN, _precision())
    assert independent and used == list(range(3, 16))
    ref2, used2, independent2 = reference(pool, list(range(16)), PLAN, _precision())
    assert not independent2 and used2 == list(range(16))   # Nothing left over: it says so.


def test_asking_for_a_wider_interval_makes_the_agent_run_longer():
    pool = markov_pool([0.3, 0.7], **FAST, seed=0)
    cheap = replay(pool, "early_stop", PLAN, _precision("standard_error"), CostModel())
    careful = replay(pool, "early_stop", PLAN, _precision("confidence"), CostModel())
    assert careful.cost["compute_ns"] > cheap.cost["compute_ns"]


def test_the_whole_comparison_is_repeated_with_different_replicas(tmp_path):
    pool = markov_pool([0.3, 0.7], **FAST, seed=1)
    out = compare_strategies(pool, PLAN, _precision(), CostModel(), repeats=4, seed=0)
    assert len(out.repeats) == 4
    assert out.repeats[0]["order"] != out.repeats[1]["order"]      # Different replicas each time.
    for name in ("fixed", "early_stop", "add_replicas", "mdagent"):
        s = out.summary[name]
        assert s["repeats"] == 4
        assert 0.0 <= s["accurate_rate"] <= 1.0
    assert out.summary["fixed"]["saved_vs_fixed"] == 0.0
    table = format_table(out)
    assert "compute ns" in table and "accurate" in table
    paths = write(out, tmp_path, "budget")
    assert [p.name for p in paths] == ["budget.json", "budget.md"]
    text = paths[1].read_text()
    assert "Markov model" in text and "cheaper only counts if it is still right" in text


def test_a_pool_that_runs_out_of_frames_says_so():
    pool = markov_pool([0.3, 0.7], rate_per_ns=0.05, n_replicas=4, ns=6.0, seed=1)
    plan = Plan(replicas=2, first_ns=2.0, extend_ns=2.0, max_ns=50.0)
    t = replay(pool, "early_stop", plan, _precision(), CostModel())
    assert t.stop_reason == "pool_exhausted" and t.ns_per_replica == 6.0


def test_an_unknown_strategy_is_refused():
    pool = markov_pool([0.3, 0.7], rate_per_ns=5.0, n_replicas=4, ns=10.0, seed=1)
    with pytest.raises(ValueError, match="unknown strategy"):
        replay(pool, "guess", PLAN, _precision(), CostModel())


def _fake_run(tmp_path, n_replicas=4, n_frames=40, hold=10, distances=None):
    """A run folder with trajectories in which one chi1 angle really changes state.

    The side chain of a valine is rotated around the CA-CB axis, so mdtraj measures the chi1
    values this test asks for, and pool_from_run has to read them back from the files.
    """
    import mdtraj as md

    from mdagent.agent import LoopSettings, RunState
    from mdagent.simulate import MDSettings

    run = tmp_path / "pool"
    run.mkdir(parents=True)
    template = md.load("examples/181L.pdb")
    val = [r for r in template.topology.residues if r.name == "VAL"][0]
    one = template.atom_slice([a.index for a in val.atoms])
    names = [a.name for a in one.topology.atoms]
    ca, cb = names.index("CA"), names.index("CB")
    moving = [names.index("CG1"), names.index("CG2")]
    one.save_pdb(str(run / "protein.pdb"))

    def rotated(delta_deg):
        xyz = one.xyz[0].copy()
        u = xyz[cb] - xyz[ca]
        u = u / np.linalg.norm(u)
        a = np.radians(delta_deg)
        for i in moving:
            v = xyz[i] - xyz[cb]
            xyz[i] = xyz[cb] + v * np.cos(a) + np.cross(u, v) * np.sin(a) + u * np.dot(u, v) * (1 - np.cos(a))
        return xyz

    start = float(np.degrees(md.compute_chi1(one)[1][0, 0]))
    frames = {}
    for target in (60.0, 180.0):
        frames[target] = rotated(target - start)
    for i in range(n_replicas):
        xyz = np.stack([frames[60.0 if (k // hold + i) % 2 == 0 else 180.0] for k in range(n_frames)])
        md.Trajectory(xyz, one.topology).save_dcd(str(run / f"replica_{i}_part000.dcd"))
    quantity = "distance_states" if distances else "chi1_populations"
    state = RunState(pdb="examples/181L.pdb", target_description="one valine", target=[("A", val.resSeq)],
                     n_replicas=n_replicas, base_seed=0, md=MDSettings(save_interval_ps=5.0),
                     loop=LoopSettings(), rounds_ns=[n_frames * 0.005], distances=distances or [],
                     goal={"quantity": quantity, "precision": 0.05})
    state.save(run)
    return run, val.resSeq


def test_a_pool_can_be_read_back_from_a_finished_run(tmp_path):
    from mdagent.budget import pool_from_run

    run, resseq = _fake_run(tmp_path, n_replicas=4, n_frames=40, hold=10)
    pool = pool_from_run(run)
    assert pool.labels == [f"VAL{resseq}"]
    assert pool.n_replicas == 4
    assert pool.ns_per_frame == 0.005 and pool.ns_per_replica == 0.2
    series = pool.series[pool.labels[0]]
    assert all(len(s) == 40 for s in series)
    assert set(np.unique(series[0])) == {0, 1}        # It really visits g+ and t.
    assert pool.truth is None                          # A real run has no known answer.
    # Half the frames in each state, and the replicas start in different states.
    assert abs((series[0] == 0).mean() - 0.5) < 0.01
    assert series[0][0] != series[1][0]


def test_a_strategy_can_be_replayed_on_a_run(tmp_path):
    from mdagent.budget import pool_from_run

    run, _ = _fake_run(tmp_path, n_replicas=6, n_frames=40, hold=5)
    pool = pool_from_run(run)
    plan = Plan(replicas=3, first_ns=0.05, extend_ns=0.05, max_ns=0.2, min_flips_per_replica=2,
                no_flip_patience_ns=0.1)
    t = replay(pool, "mdagent", plan, Precision(epsilon=0.2, looks=plan.looks()), CostModel(equilibration_ns=0.0))
    assert t.cost["aggregate_ns"] > 0 and t.stop_reason in ("target_reached", "budget_exhausted")
    assert set(t.results) == set(pool.labels)


def test_a_distance_pool_can_be_read_back_too(tmp_path):
    """The other observable: the distance between two atoms, with a near and a far state."""
    from mdagent.budget import pool_from_run

    _, resseq = _fake_run(tmp_path / "find", n_replicas=2, n_frames=10)   # Find the residue number.
    spec = f"A:{resseq}:CG1-A:{resseq}:N@0.3"
    run, _ = _fake_run(tmp_path, n_replicas=3, n_frames=40, hold=10, distances=[spec])
    pool = pool_from_run(run)
    assert pool.labels == [spec]
    assert pool.state_names[pool.labels[0]] == ["near", "far"]
    assert all(len(s) == 40 for s in pool.series[pool.labels[0]])
