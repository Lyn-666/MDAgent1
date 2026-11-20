"""Experiment A: backbone basins, the four arms, and the verdict. Models with known answers, no GPU."""

import json

import mdtraj as md
import numpy as np
import pytest

from mdagent.basins import BasinSet, angles, assign, dihedral_atoms, parse_sets, torus_distance
from mdagent.experiment_a import (Design, Targets, allocate, compare_to_reference, estimate, judge_online,
                                  model_pool, reference, run, select_items, write_outputs, z_bonferroni)

FAST = [([0.7, 0.3], 8.0)]
SMALL = Design(families=(1, 4), sequential=(4,), max_ns=40.0, budget_ns=320.0, scan_ns=10.0)


# --- basins --------------------------------------------------------------------------------------

def test_observables_are_written_like_rama_and_turn():
    rama, turn = parse_sets("rama:A:73; turn:A:52")
    assert rama.names == ["alphaR", "beta", "alphaL"] and rama.dihedrals == [(0, "phi"), (0, "psi")]
    assert turn.names == ["betaI", "betaII"] and turn.dihedrals == [(0, "psi"), (1, "phi")]   # psi D52, phi G53
    assert [s.label for s in parse_sets("rama:A:2-4")] == ["rama:A:2", "rama:A:3", "rama:A:4"]
    with pytest.raises(ValueError, match="rama:A:73"):
        BasinSet.parse("ramachandran 73")


def test_angles_wrap_around():
    assert torus_distance(np.array([[179.0, 0.0]]), (-179.0, 0.0))[0] == pytest.approx(2.0)


def test_a_frame_between_basins_keeps_the_last_basin():
    """Transition-based assignment: recrossing a boundary is not a transition."""
    turn = BasinSet.parse("turn:A:52")
    path = np.array([[-30, -90], [40, 0], [-30, -90], [40, 0], [150, 90], [100, 30], [150, 90]], float)
    assert assign(path, turn).tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_the_dihedrals_are_the_ones_mdtraj_computes():
    t = md.load("examples/181L.pdb")
    rama = BasinSet.parse("rama:A:52")
    phi_i, phi = md.compute_phi(t)
    psi_i, psi = md.compute_psi(t)
    want_phi = np.degrees(phi[0][[k for k, q in enumerate(phi_i) if t.topology.atom(int(q[1])).residue.resSeq == 52][0]])
    want_psi = np.degrees(psi[0][[k for k, q in enumerate(psi_i) if t.topology.atom(int(q[1])).residue.resSeq == 52][0]])
    got = angles(t, dihedral_atoms(t.topology, rama))[0]
    assert got == pytest.approx([want_phi, want_psi], abs=1e-3)


# --- estimates -----------------------------------------------------------------------------------

def test_estimates_recover_the_model():
    pool = model_pool(FAST, n_replicas=8, ns=60.0, seed=1)
    e = estimate(pool, "model1", list(range(8)), 60.0, Design())
    assert e["populations"]["betaI"] == pytest.approx(0.7, abs=0.03)
    # Leaving betaI for betaII happens at (jump rate) x p(betaII) = 8 x 0.3 per ns.
    assert e["rates"]["betaI->betaII"]["k_per_ns"] == pytest.approx(2.4, rel=0.15)
    assert e["dg_kt"] == pytest.approx(-np.log(0.3 / 0.7), abs=0.15)


def test_trapped_replicas_are_never_called_converged():
    """All replicas in one basin give a tiny error; that must not count as precise."""
    pool = model_pool([([0.96, 0.04], 0.0001)], n_replicas=6, ns=20.0, seed=0)
    e = estimate(pool, "model1", list(range(6)), 20.0, Design())
    verdict = judge_online(e, 2.0, Targets())
    assert not verdict["adequate"] and not verdict["populations"] and not verdict["landscape"]


def test_the_offline_check_measures_the_real_error_and_the_coverage():
    pool = model_pool(FAST, n_replicas=12, ns=40.0, seed=2)
    ref = reference(pool, list(range(6, 12)), Design(), Targets())
    e = estimate(pool, "model1", [0, 1, 2], 40.0, Design())
    c = compare_to_reference(e, ref["model1"], 2.0, Targets())
    assert c["population_error"] == pytest.approx(abs(e["populations"]["betaI"] - ref["model1"]["populations"]["betaI"]))
    e["populations"] = {"betaI": 0.2, "betaII": 0.8}                    # a wrong answer
    assert not compare_to_reference(e, ref["model1"], 2.0, Targets())["populations"]


def test_more_replicas_only_when_the_spread_between_them_is_real():
    t = Targets()
    calm = {"x": {"within": 0.03, "between": 0.03}}
    split = {"x": {"within": 0.01, "between": 0.025}}
    hopeless = {"x": {"within": 0.01, "between": 0.06}}
    m, length, _ = allocate(calm, ["x"], 4, 10.0, 2.4, t, Design(), 8, 100.0, 0.6)
    assert m == 4 and length > 10.0                       # run longer: cheaper than new equilibrations
    m2, _, _ = allocate(split, ["x"], 4, 10.0, 2.4, t, Design(), 8, 100.0, 0.6)
    assert m2 > 4                                          # the replicas disagree: only more replicas help
    assert allocate(hopeless, ["x"], 4, 10.0, 2.4, t, Design(), 8, 100.0, 0.6) is None   # not even 8 would do


def test_fewer_looks_allow_a_narrower_calibrated_interval():
    assert z_bonferroni(0.05, 1) == pytest.approx(1.96, abs=0.01)
    assert z_bonferroni(0.05, 3) < z_bonferroni(0.05, 20)


# --- the experiment ------------------------------------------------------------------------------

def test_the_experiment_runs_every_arm_and_writes_its_report(tmp_path):
    pool = model_pool(FAST, n_replicas=12, ns=40.0, seed=0)
    out = run(pool, SMALL, Targets(), trials=4, holdout=4, seed=0)
    assert set(out["arms"]) == {"sequential_ci_4", "agent"}
    assert set(out["summary"]["fixed"]) == {"1", "4"}
    for t in out["arms"]["agent"]:
        assert t["decision"] in ("STOP", "INSUFFICIENT_EVIDENCE")
        assert t["outcome"] in ("correct_stop", "false_stop", "correct_insufficient", "over_cautious")
        assert t["cost"]["compute_ns"] > 0 and t["looks"] <= SMALL.agent_looks + 1   # + the planning look
        assert t["steps"][0]["planning"]                                           # the first look only plans
    assert out["verdict"]["result"] in ("supported", "not_supported", "reference_inadequate")
    assert set(out["verdict"]["vs_sequential_ci"]) == {"sequential_ci_4"}
    names = [p.name for p in write_outputs(out, tmp_path)]
    assert names == ["experiment_A.json", "experiment_A.md", "experiment_A_cost_accuracy.csv",
                     "experiment_A_cost_accuracy.svg"]
    json.loads((tmp_path / "experiment_A.json").read_text())                       # valid JSON, no NaN
    md_text = (tmp_path / "experiment_A.md").read_text()
    assert "## Verdict" in md_text and "Sequential-CI" in md_text and "C_hit" in md_text
    assert "<svg" in (tmp_path / "experiment_A_cost_accuracy.svg").read_text()


def test_held_out_replicas_are_never_used_by_a_method():
    pool = model_pool(FAST, n_replicas=12, ns=40.0, seed=0)
    out = run(pool, SMALL, Targets(), trials=3, holdout=4, seed=5)
    assert len(out["pool"]["held_out"]) == 4


def test_a_trapped_reference_cannot_judge_anyone():
    """A-Hard: if even the reference never crosses, no accuracy claim can be made."""
    pool = model_pool([([0.96, 0.04], 0.0001)], n_replicas=12, ns=40.0, seed=0)
    out = run(pool, SMALL, Targets(), trials=3, holdout=4, seed=0)
    assert out["verdict"]["result"] == "reference_inadequate"
    assert all(t["decision"] == "INSUFFICIENT_EVIDENCE" for t in out["arms"]["agent"])


def test_the_pilot_picks_items_that_plain_md_can_answer():
    pool = model_pool([([0.7, 0.3], 8.0), ([0.96, 0.04], 0.0001), ([0.5, 0.5], 4.0)], n_replicas=3, ns=10.0)
    chosen, counts = select_items(pool, at_most=5)
    assert chosen == ["model1", "model3"] and min(counts["model2"]) < 4
    assert select_items(pool, at_most=1)[0] == ["model3"]    # the slowest of the fast ones is kept


def test_basins_are_read_from_the_trajectories_of_a_run(tmp_path):
    """The path the real pool takes: protein.pdb, replica DCDs and run.json -> angles and basin states."""
    from mdagent.agent import LoopSettings, RunState
    from mdagent.experiment_a import pool_from_run
    from mdagent.simulate import MDSettings

    t = md.load("examples/181L.pdb")
    keep = [a.index for a in t.topology.atoms if a.residue.chain.chain_id == "A" and 50 <= a.residue.resSeq <= 54]
    part = t.atom_slice(keep)
    part.save_pdb(str(tmp_path / "protein.pdb"))
    rng = np.random.default_rng(0)
    for i in range(3):
        xyz = part.xyz[0] + rng.normal(0, 0.002, (20, part.n_atoms, 3)).astype(np.float32)
        md.Trajectory(xyz, part.topology).save_dcd(str(tmp_path / f"replica_{i}_part000.dcd"))
    RunState(pdb="x", target_description="", target=[("A", 52)], n_replicas=3, base_seed=0,
             md=MDSettings(save_interval_ps=5.0), loop=LoopSettings(), rounds_ns=[0.1]).save(tmp_path)
    pool = pool_from_run(tmp_path, "rama:A:52;turn:A:52")
    assert pool.labels == ["rama:A:52", "turn:A:52"] and pool.n_replicas == 3
    assert pool.ns_per_frame == 0.005 and pool.values["turn:A:52"][0].shape == (20, 2)
    static = angles(part, dihedral_atoms(part.topology, BasinSet.parse("turn:A:52")))[0]
    assert np.allclose(pool.values["turn:A:52"][0].mean(axis=0), static, atol=3.0)
