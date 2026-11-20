"""Phase 2: OpenMM adapters (featurizers, tool views) and the shadow mode, on fake runs with known states.

A fake run is the test peptide with the chi1 angles of Val111 and Phe114 set frame by frame, so the state
sequences are known. Fake metadynamics data are chi1 samples written as enhanced/replica_<r>_chi.npy.
No simulation runs here.
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdagent import shadow
from mdagent.agent import Goal, LoopSettings, RunState
from mdagent.simulate import MDSettings
from mdagent.tools_openmm import MetadTool, PlainMDTool, featurize

DATA = Path(__file__).resolve().parent / "data"
CENTER = {0: 60.0, 1: 180.0, 2: -60.0}   # g+, t, g-
TARGET = [("A", 111), ("A", 114)]


def _rotator(top):
    """For each target residue: (chi1 atom indices, atoms to rotate about CA-CB)."""
    out = {}
    idx, _ = md.compute_chi1(md.load(str(DATA / "fragment_108-114.pdb")))
    for quad in idx:
        res = top.atom(int(quad[1])).residue
        if (res.chain.chain_id, res.resSeq) in TARGET:
            fixed = {"N", "CA", "C", "O", "H", "HA", "CB"}
            out[res.resSeq] = (quad, [a.index for a in res.atoms if a.name not in fixed])
    return out


def _set_chi(xyz, quad, moving, target_deg):
    ca, cb = xyz[quad[1]], xyz[quad[2]]
    axis = (cb - ca) / np.linalg.norm(cb - ca)

    def chi(x):
        return np.degrees(md.compute_dihedrals(md.Trajectory(x[None], None), [quad])[0, 0])

    def rotate(x, deg):
        th = np.radians(deg)
        v = x[moving] - cb
        x = x.copy()
        x[moving] = (cb + v * np.cos(th) + np.cross(axis, v) * np.sin(th)
                     + np.outer(v @ axis, axis) * (1 - np.cos(th)))
        return x
    delta = (target_deg - chi(xyz) + 180) % 360 - 180
    out = rotate(xyz, delta)
    if abs((chi(out) - target_deg + 180) % 360 - 180) > 1:
        out = rotate(xyz, -delta)
    return out


def _states(rng, n, stay=0.9, start=0):
    s = [start]
    for _ in range(n - 1):
        s.append(s[-1] if rng.random() < stay else int(rng.integers(0, 3)))
    return np.array(s)


def make_run(tmp_path, rounds=(0.1, 0.1, 0.1), n_replicas=2, frames_per_round=20, stay=(0.9, 0.9), seed=0,
             goal=None, **state_kw):
    run_dir = Path(tmp_path)
    shutil.copy(DATA / "fragment_108-114.pdb", run_dir / "protein.pdb")
    t0 = md.load(str(run_dir / "protein.pdb"))
    rot = _rotator(t0.topology)
    rng = np.random.default_rng(seed)
    truth = {}
    for r in range(n_replicas):
        seqs = {k: _states(rng, frames_per_round * len(rounds), stay[i]) for i, k in enumerate(sorted(rot))}
        truth[r] = seqs
        for p in range(len(rounds)):
            frames = []
            for f in range(p * frames_per_round, (p + 1) * frames_per_round):
                x = t0.xyz[0]
                for k, (quad, moving) in rot.items():
                    x = _set_chi(x, quad, moving, CENTER[int(seqs[k][f])] + rng.normal(0, 5))
                frames.append(x)
            md.Trajectory(np.array(frames), t0.topology).save_dcd(str(run_dir / f"replica_{r}_part{p:03d}.dcd"))
    state = RunState(pdb="x", target_description="test", target=list(TARGET), n_replicas=n_replicas, base_seed=1,
                     md=MDSettings(save_interval_ps=5.0), loop=LoopSettings(first_ns=0.1, extend_ns=0.1, max_ns=0.3),
                     rounds_ns=list(rounds), goal=asdict(goal or Goal.for_mode("auto", precision=0.1)), **state_kw)
    state.save(run_dir)
    return run_dir, state, truth


def test_featurize_recovers_the_chi1_states_frame_by_frame(tmp_path):
    run_dir, state, truth = make_run(tmp_path)
    data = featurize(run_dir, state, n_parts=3, n_replicas=2)
    assert set(data) == {"VAL111", "PHE114"}
    for r in range(2):
        assert np.array_equal(data["VAL111"].states[r], truth[r][111])
        assert np.array_equal(data["PHE114"].states[r], truth[r][114])


def test_a_view_of_an_earlier_moment_sees_only_that_data(tmp_path):
    run_dir, state, _ = make_run(tmp_path)
    view = PlainMDTool(run_dir, state, parts=1, n_replicas=1)
    assert view.length() == pytest.approx(0.1) and view.replicas() == 1
    d = view.observe(["VAL111"])["VAL111"]
    assert len(d.states) == 1 and len(d.states[0]) == 20
    assert view.supports("state_populations", "VAL111") and not view.supports("event", "VAL111")
    with pytest.raises(RuntimeError):
        view.advance(1, 0.2)


def test_task_mapping_keeps_the_users_limits_and_permissions(tmp_path):
    goal = Goal.for_mode("auto", precision=0.07)
    goal.permissions = ["stop_early", "extend", "switch_method"]
    run_dir, state, _ = make_run(tmp_path, goal=goal,
                                 enhanced={"enabled": True, "settings": {"length_ns": 2.0, "n_replicas": 2}})
    task = shadow.task_from_state(run_dir, state, PlainMDTool(run_dir, state).kinds())
    assert [r.estimand for r in task.requirements] == ["VAL111", "PHE114"]
    assert all(r.params["max_se"] == 0.07 for r in task.requirements)
    assert task.budget.planned_cost == pytest.approx(0.3 * 2 + 2.0 * 2)    # Plain MD plus enhanced sampling.
    assert task.budget.max_cost == pytest.approx(2 * task.budget.planned_cost)
    assert task.allowed_methods == ("metadynamics",) and "switch_method" in task.granted()
    state.mode = "guided"
    assert "switch_method" not in shadow.task_from_state(run_dir, state, {"VAL111": "state_populations"}).granted()


def test_replay_asks_the_core_at_every_round_and_compares_with_the_policy(tmp_path):
    run_dir, state, _ = make_run(tmp_path)
    state.notes["plain_md_outcome"] = {"reason": "budget_exhausted", "ns_per_replica": 0.3}
    state.save(run_dir)
    records = shadow.replay(run_dir, log=lambda line: None)
    assert [r["ns_per_replica"] for r in records] == pytest.approx([0.1, 0.2, 0.3])
    assert all(r["phase"] == "plain_md" for r in records)
    assert (run_dir / shadow.REPLAY_FILE).exists()
    table, counts = shadow.compare(run_dir, records)
    assert "continue" in table and "budget_exhausted" in table
    assert sum(counts.values()) == 3
    again = shadow.replay(run_dir, log=lambda line: None)       # The file is rewritten, with the same decisions.
    assert [r["decision"]["chosen"]["kind"] for r in again] == [r["decision"]["chosen"]["kind"] for r in records]
    assert len(shadow.load(run_dir / shadow.REPLAY_FILE)) == 3


def test_old_actions_and_replica_counts_come_from_run_json(tmp_path):
    run_dir, state, _ = make_run(tmp_path, n_replicas=4)
    state.notes["actions"] = [{"phase": "plain_md", "action": "add_replicas", "executed": True, "ns_per_replica": 0.1,
                               "reason": "The replicas clearly disagree. Now 4 replicas.", "time": ""}]
    state.notes["plain_md_outcome"] = {"reason": "target_reached", "ns_per_replica": 0.3}
    assert shadow.first_replicas(state) == 2
    assert [shadow.replicas_at(state, "plain_md", ns) for ns in (0.1, 0.2)] == [2, 4]
    assert shadow.old_action(run_dir, state, "plain_md", 0.1) == ("add_replicas", "")
    assert shadow.old_action(run_dir, state, "plain_md", 0.2) == ("continue", "")
    assert shadow.old_action(run_dir, state, "plain_md", 0.3) == ("stop", "target_reached")
    state.notes["first_n_replicas"] = 3
    assert shadow.first_replicas(state) == 3


def _fake_metad(run_dir, state, n_samples=400, n_replicas=2, seed=0):
    from mdagent.enhanced import MetadSettings

    s = MetadSettings(length_ns=n_samples / 1000, n_replicas=n_replicas, pace_ps=1.0, grid_points=37, chunk_ns=0.1,
                      height_kj=0.5)
    (run_dir / "enhanced").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for r in range(n_replicas):
        chi = np.radians([CENTER[int(x)] + rng.normal(0, 10) for x in _states(rng, n_samples, 0.95)])[:, None]
        np.save(run_dir / "enhanced" / f"replica_{r}_chi.npy", chi)
        (run_dir / "enhanced" / f"replica_{r}.json").write_text(json.dumps({"ns_done": n_samples / 1000}))
    state.enhanced = {"enabled": True, "requested": True, "settings": asdict(s), "residues": [["A", 111]],
                      "resnames": {"A:111": "VAL"}, "outcome": {"reason": "target_reached",
                                                                  "ns_per_replica": n_samples / 1000}}
    state.save(run_dir)


def test_metadynamics_view_is_causal_so_a_replay_matches_the_past(tmp_path):
    run_dir, state, _ = make_run(tmp_path)
    _fake_metad(run_dir, state)
    full = MetadTool(run_dir, state)
    assert full.length() == pytest.approx(0.4) and full.supports("state_populations", "VAL111")
    assert not full.supports("state_populations", "PHE114")       # Not biased.
    early = MetadTool(run_dir, state, until_ns=0.2).observe(["VAL111"])["VAL111"]

    # The same run as it was at 0.2 ns: files cut at 200 samples.
    past = tmp_path / "past"
    past.mkdir()
    shutil.copy(run_dir / "protein.pdb", past / "protein.pdb")
    (past / "enhanced").mkdir()
    for r in range(2):
        np.save(past / "enhanced" / f"replica_{r}_chi.npy", np.load(run_dir / "enhanced" / f"replica_{r}_chi.npy")[:200])
        (past / "enhanced" / f"replica_{r}.json").write_text(json.dumps({"ns_done": 0.2}))
    then = MetadTool(past, state).observe(["VAL111"])["VAL111"]
    for a, b in zip(early.states, then.states):
        assert np.array_equal(a, b)
    for a, b in zip(early.weights, then.weights):
        assert np.allclose(a, b)


def test_replay_covers_enhanced_sampling_checks_with_the_reweighted_estimator(tmp_path):
    run_dir, state, _ = make_run(tmp_path)
    state.notes["plain_md_outcome"] = {"reason": "stalled", "ns_per_replica": 0.3}
    _fake_metad(run_dir, state)
    records = shadow.replay(run_dir, log=lambda line: None)
    enhanced = [r for r in records if r["phase"] == "enhanced_sampling"]
    assert [r["ns_per_replica"] for r in enhanced] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    val = [a for a in enhanced[-1]["assessments"] if a["estimand"] == "VAL111"][0]
    assert val["estimator"] == "populations.reweighted"
    phe = [a for a in enhanced[-1]["assessments"] if a["estimand"] == "PHE114"][0]
    assert phe["estimator"] == "populations.unbiased_time_fraction"     # Still from plain MD.
    table, _ = shadow.compare(run_dir, records)
    assert "switch_method (stalled)" in table and "stop (target_reached)" in table


def test_live_shadow_records_and_never_breaks_the_run(tmp_path, monkeypatch):
    run_dir, state, _ = make_run(tmp_path)
    lines = []
    rec = shadow.record(run_dir, state, "plain_md", log=lines.append)
    assert rec["phase"] == "plain_md" and rec["shadow"] and "not acted on" in lines[-1]
    assert shadow.load(run_dir / shadow.LIVE_FILE)[-1]["kind"] == "decision"

    (run_dir / "protein.pdb").unlink()                                      # Something goes wrong.
    assert shadow.record(run_dir, state, "plain_md", log=lines.append) is None
    assert shadow.load(run_dir / shadow.LIVE_FILE)[-1]["kind"] == "shadow_error"
    assert "skipped" in lines[-1]

    monkeypatch.setenv("MDAGENT_SHADOW", "0")
    n = len(shadow.load(run_dir / shadow.LIVE_FILE))
    assert shadow.record(run_dir, state, "plain_md", log=lines.append) is None
    assert len(shadow.load(run_dir / shadow.LIVE_FILE)) == n


def test_the_running_agent_writes_a_shadow_decision_after_its_own(tmp_path):
    import mdagent.agent as agent

    run_dir, state, _ = make_run(tmp_path)
    reports, decisions, next_ns = agent._diagnose_and_decide(run_dir, state)
    live = shadow.load(run_dir / shadow.LIVE_FILE)
    assert len(live) == 1 and live[0]["ns_per_replica"] == pytest.approx(0.3)
    assert {a["estimand"] for a in live[0]["assessments"]} == {r.label for r in reports if r.transitions}
    agent._diagnose_and_decide(run_dir, RunState.load(run_dir))     # Same round again: no second record.
    assert len(shadow.load(run_dir / shadow.LIVE_FILE)) == 1


def test_distance_and_event_runs_map_to_their_estimands(tmp_path):
    run_dir, state, _ = make_run(tmp_path)
    state.goal["quantity"] = "distance_states"
    state.distances = ["A:111:CB-A:114:CZ@0.6"]
    data = featurize(run_dir, state, 3, 2)
    assert list(data) == ["A:111:CB-A:114:CZ@0.6"] and data["A:111:CB-A:114:CZ@0.6"].names == ["near", "far"]
    state.goal["quantity"] = "event_seen"
    state.events = ["A:111=t"]
    task = shadow.task_from_state(run_dir, state, PlainMDTool(run_dir, state).kinds())
    assert task.requirements[0].kind == "event" and task.requirements[0].params["hold_frames"] == 20


@pytest.mark.slow
def test_plain_md_tool_runs_rounds_and_lets_new_replicas_catch_up(tmp_path):
    """A tiny peptide on CPU: the tool wraps run_segment exactly like agent.run_loop does."""
    from mdagent.build import SetupOptions, build_system
    from mdagent.simulate import SystemFiles, part_path, write_inputs

    built = build_system(DATA / "fragment_108-114.pdb", SetupOptions(forcefield="amber14", padding_nm=0.8))
    write_inputs(built, tmp_path)
    state = RunState(pdb="x", target_description="t", target=list(TARGET), n_replicas=2, base_seed=1,
                     md=MDSettings(equilibration_ps=1.0, save_interval_ps=0.5, platform="CPU"),
                     loop=LoopSettings(first_ns=0.002, extend_ns=0.002, max_ns=0.004))
    state.save(tmp_path)
    tool = PlainMDTool(tmp_path, state, files=SystemFiles(built.topology, built.positions, built.system),
                       log=lambda line: None)
    tool.advance(2, 0.002)
    tool.advance(3, 0.002)            # A new replica catches up; no new round.
    assert RunState.load(tmp_path).rounds_ns == [0.002] and part_path(tmp_path, 2, 0, ".state.xml").exists()
    tool.advance(3, 0.004)
    saved = RunState.load(tmp_path)
    assert saved.rounds_ns == [0.002, 0.002] and saved.n_replicas == 3 and saved.pending_round_ns == 0
    data = tool.observe(["VAL111"])["VAL111"]
    assert [len(s) for s in data.states] == [8, 8, 8]
