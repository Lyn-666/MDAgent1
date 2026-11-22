"""Backbone phi/psi basins, transition statistics and free-energy landscapes, for any residue.

Numerics are checked on synthetic phi/psi series with known switching probabilities and basin populations;
the pipeline is checked on a fake run of the test peptide (see test_shadow.make_run). No MD runs here.
"""

import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from mdagent.backbone import (KB_KCAL, BackboneSettings, assign_basins, backbone_quads, core_index,
                              diagnose_backbone, landscape, parse_basins, transition_statistics, worst_landscape_error,
                              worst_rate_error)
from mdagent.estimators import AngleData, FreeEnergySurface, StateData, TransitionRates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shadow import make_run  # noqa: E402

CENTERS = {0: (-65.0, -40.0), 1: (-120.0, 135.0)}   # inside alpha_R and beta of the default basins
DT = 0.001   # ns per frame


def series(rng, n, p01, p10, sd=12.0, start=0):
    s = [start]
    for _ in range(n - 1):
        x = s[-1]
        s.append(1 - x if rng.random() < (p01 if x == 0 else p10) else x)
    s = np.array(s)
    phi = np.array([CENTERS[x][0] for x in s]) + rng.normal(0, sd, n)
    psi = np.array([CENTERS[x][1] for x in s]) + rng.normal(0, sd, n)
    return s, phi, psi


def test_basins_are_data_with_a_default_and_wrap_around_180():
    default = parse_basins(None)
    assert [b.name for b in default] == ["alpha_R", "beta", "alpha_L", "other"]
    pts = [(-60, -45), (-120, 130), (-170, -170), (60, 45), (80, 180), (-60, 50)]
    got = core_index([p[0] for p in pts], [p[1] for p in pts], default)
    assert [default[i].name if i >= 0 else None for i in got] == ["alpha_R", "beta", "beta", "alpha_L", "other", None]
    own = parse_basins("a:-180,0,-180,180; b:0,180,-180,180; b:-10,10,0,10")
    assert [b.name for b in own] == ["a", "b"] and len(own[1].boxes) == 2
    for bad in ["a:1,2,3", "x", "only:-180,0,0,180"]:
        with pytest.raises(ValueError):
            parse_basins(bad)


def test_frames_between_cores_keep_the_last_basin():
    basins = parse_basins(None)
    phi = [-65, -65, -60, -120, -120]
    psi = [-40, 45, 55, 130, 45]       # psi 45 and 55 are within the 10 degree margin of the boundary at 50.
    assert list(assign_basins(phi, psi, basins)) == [0, 0, 0, 1, 1]


def test_transition_rates_and_dwell_times_match_the_known_process():
    p01, p10 = 0.004, 0.012
    z_rate, z_dwell = [], []
    for seed in range(30):     # A calibration test: no bias, and about 95 % within two standard errors.
        rng = np.random.default_rng(seed)
        states = [series(rng, 10000, p01, p10)[0] for _ in range(4)]
        st = transition_statistics(states, ["alpha_R", "beta"], DT)
        k01, e01 = st["rates_per_ns"]["alpha_R"]["beta"]
        z_rate.append((k01 - p01 / DT) / e01)
        mean, se, n = st["mean_dwell_ns"]["alpha_R"]
        z_dwell.append((mean - DT / p01) / se)
        assert st["total_transitions"] == sum(sum(r) for r in st["counts"])
    for z in (np.array(z_rate), np.array(z_dwell)):
        assert abs(z.mean()) < 0.6 and np.mean(np.abs(z) <= 2) >= 0.8


def test_rate_error_uses_leading_transitions_and_never_seen_is_infinite():
    st = {"names": ["a", "b", "c"], "counts": [[0, 16, 0], [16, 0, 0], [0, 0, 0]], "time_ns": [5, 5, 0]}
    worst, lead = worst_rate_error(st, {"a": 0.5, "b": 0.5, "c": 0.0})
    assert worst == pytest.approx(0.25) and lead == ["a->b", "b->a"]
    st["counts"][1][0] = 0
    assert math.isinf(worst_rate_error(st, {"a": 0.5, "b": 0.5, "c": 0.0})[0])
    worst, lead = worst_rate_error({"names": ["a", "b"], "counts": [[0, 4], [4, 0]], "time_ns": [9, 1]},
                                   {"a": 0.98, "b": 0.02})
    assert lead == ["a->any"] and worst == pytest.approx(0.5)   # One leading basin: the rate of leaving it.


def test_landscape_recovers_the_basin_free_energy_difference():
    rng = np.random.default_rng(2)
    runs = [series(rng, 20000, 0.01, 0.03) for _ in range(4)]   # Stationary: 3/4 in alpha_R, 1/4 in beta.
    fes = landscape([r[1] for r in runs], [r[2] for r in runs], 300.0, bins_deg=15.0)
    p = fes["probability"]
    basins = parse_basins(None)
    e = fes["edges_deg"]
    c = (e[:-1] + e[1:]) / 2
    phi, psi = np.meshgrid(c, c, indexing="ij")
    which = assign_basins(phi.ravel(), psi.ravel(), basins).reshape(p.shape)
    ratio = p[which == 0].sum() / p[which == 1].sum()
    assert ratio == pytest.approx(3.0, rel=0.15)
    assert np.nanmin(fes["free_energy_kcal"]) == 0.0
    assert worst_landscape_error(fes, 2.0) < 0.5
    one = landscape([runs[0][1]], [runs[0][2]], 300.0)
    assert math.isinf(worst_landscape_error(one, 2.0))      # One replica: no spread, no error estimate.


def _angles(n_rep=4, n=20000, p01=0.01, p10=0.03, seed=3):
    rng = np.random.default_rng(seed)
    runs = [series(rng, n, p01, p10) for _ in range(n_rep)]
    from mdagent.basins import BasinSet
    return {"ALA5": (BasinSet.parse("rama:A:5"), [np.column_stack([r[1], r[2]]) for r in runs])}


def test_verdict_needs_populations_rates_and_landscape_together():
    a = _angles()
    ok, details = diagnose_backbone(a, BackboneSettings(), 20.0, DT, 300.0, precision=0.05, rate_rel_error=0.3,
                                    fes_error_kcal=0.6)
    assert ok[0].verdict == "ok", ok[0].reason
    assert {"transitions", "landscape", "minima", "basins", "axes"} <= set(details["ALA5"])
    assert ok[0].extra["dg_kt"] == pytest.approx(math.log(3), abs=0.15)   # From basins.summarize: ln(0.75 / 0.25).
    assert details["ALA5"]["minima"]["alpha_R"][0] == pytest.approx(0.0, abs=0.01)
    strict_rate = diagnose_backbone(a, BackboneSettings(), 20.0, DT, 300.0, precision=0.05, rate_rel_error=0.01)[0][0]
    assert strict_rate.verdict == "imprecise" and "transition rates" in strict_rate.reason
    strict_fes = diagnose_backbone(a, BackboneSettings(), 20.0, DT, 300.0, precision=0.05, rate_rel_error=0.3,
                                   fes_error_kcal=0.001)[0][0]
    assert strict_fes.verdict == "imprecise" and "landscape" in strict_fes.reason


def test_a_residue_that_never_changes_basin_is_not_converged():
    a = _angles(p01=0.0, p10=0.0)
    rep = diagnose_backbone(a, BackboneSettings(), 20.0, DT, 300.0, precision=0.5, rate_rel_error=0.9,
                            fes_error_kcal=10)[0][0]
    assert rep.verdict == "no_flip"


def test_core_estimators_for_rates_and_landscape():
    spec, values = _angles()["ALA5"]
    a = (None, None, [v[:, 0] for v in values], [v[:, 1] for v in values])
    basins = parse_basins(None)
    states = StateData([assign_basins(f, s, basins) for f, s in zip(a[2], a[3])], [b.name for b in basins])
    r = TransitionRates().assess("ALA5 transitions", states, {"max_rel_error": 0.3, "dt_ns": DT})
    assert r.passes and r.signals["uncertainty_ratio"] < 1
    assert TransitionRates().assess("x", states, {"max_rel_error": 0.01}).status == "not_met"
    f = FreeEnergySurface().assess("ALA5 landscape", AngleData(a[2], a[3], states, DT), {"max_error_kcal": 0.6})
    assert f.passes
    single = AngleData(a[2][:1], a[3][:1], StateData(states.states[:1], states.names), DT)
    assert not FreeEnergySurface().assess("x", single, {"max_error_kcal": 9}).passes   # Needs two replicas.
    frozen = StateData([np.zeros(100, dtype=int)] * 2, states.names)
    assert TransitionRates().assess("x", frozen, {"max_rel_error": 0.5}).status == "indeterminate"


def test_chain_ends_have_no_phi_or_psi():
    import mdtraj as md

    top = md.load_topology(str(Path(__file__).resolve().parent / "data" / "fragment_108-114.pdb"))
    found, skipped = backbone_quads(top, [("A", n) for n in range(108, 115)])
    assert list(found) == ["THR109", "GLY110", "VAL111", "ALA112", "GLY113"]
    assert [k for k, _ in skipped] == [("A", 108), ("A", 114)] and "phi" in skipped[0][1] and "psi" in skipped[1][1]


def test_a_backbone_run_is_diagnosed_reported_and_shadowed(tmp_path):
    from mdagent import report, shadow
    from mdagent.agent import Goal, RunState, diagnose_run
    from mdagent.outputs import output_files
    from mdagent.tools_openmm import PlainMDTool

    run_dir, state, _ = make_run(tmp_path, goal=Goal.for_mode("auto", quantity="backbone_basins", precision=0.1))
    state.target = [("A", n) for n in range(108, 115)]
    state.backbone = ["rama:A:108-114"]
    state.backbone_settings = {"margin_deg": 10.0}
    state.save(run_dir)
    reports = diagnose_run(run_dir, state)
    verdicts = {r.label: r.verdict for r in reports}
    assert verdicts["rama:A:111"] == "no_flip"           # The fake run moves side chains only.
    assert sum(v == "no_phi_psi" for v in verdicts.values()) == 2   # 108 has no phi, 114 no psi.
    files = {str(p) for p in output_files(run_dir)}
    assert {"backbone/rama_A_111.json", "backbone/rama_A_111_fes.csv", "backbone/rama_A_111_fes.npz"} <= files

    kinds = PlainMDTool(run_dir, state).kinds()
    assert kinds["rama:A:111 basins"] == "state_populations" and kinds["rama:A:111 transitions"] == "transition_rates"
    assert kinds["rama:A:111 landscape"] == "free_energy_surface" and len(kinds) == 15
    rec = shadow.record(run_dir, state, "plain_md", log=lambda line: None)
    assert rec is not None
    by = {a["estimand"]: a for a in rec["assessments"]}
    assert by["rama:A:111 landscape"]["estimator"] == "landscape.replica_spread"
    assert all(a["status"] == "indeterminate" for a in by.values())   # No basin change: never "met".

    from mdagent.agent import _record, decide
    decisions, _ = decide(reports, state.ns_done, state.loop)
    _record(run_dir, RunState.load(run_dir), reports, decisions, 0.0)
    zh, en = report.write(run_dir)
    text = en.read_text()
    assert "backbone basins (rama:A:108-114)" in text and "Transitions between basins" in text


def test_older_backbone_runs_of_both_branches_keep_their_meaning(tmp_path):
    import json

    from mdagent.agent import Goal, RunState, diagnose_run

    run_dir, state, _ = make_run(tmp_path, goal=Goal.for_mode("auto", quantity="backbone_basins"))
    raw = json.loads((run_dir / "run.json").read_text())
    # This branch before the merge: "backbone" held the basin settings, every target residue was measured.
    mine = {**raw, "target": [["A", 110], ["A", 111]], "backbone": {"margin_deg": 12.0}}
    mine.pop("backbone_settings")
    (run_dir / "run.json").write_text(json.dumps(mine))
    s = RunState.load(run_dir)
    assert s.backbone == ["rama:A:110", "rama:A:111"] and s.backbone_settings == {"margin_deg": 12.0}
    assert {r.label for r in diagnose_run(run_dir, s)} == {"rama:A:110", "rama:A:111"}
    # The other branch before the merge: a list of items and no settings: its rama basins were three centres.
    theirs = {**raw, "backbone": ["rama:A:111"]}
    theirs.pop("backbone_settings")
    (run_dir / "run.json").write_text(json.dumps(theirs))
    s = RunState.load(run_dir)
    assert s.backbone_settings == {"rama": "centres"}
    rep = [r for r in diagnose_run(run_dir, s) if r.label == "rama:A:111"][0]
    assert list(rep.populations) == ["alphaR", "beta", "alphaL"]


def test_the_chat_form_turns_a_backbone_request_into_a_run():
    from mdagent.cli import build_parser
    from mdagent.llm import RunPlan, to_argv

    plan = RunPlan(target_kind="backbone", residues="A:110-112", replicas=3, max_replicas=8, max_ns=50.0,
                   allow="stop_early,add_replicas", questions=[], summary="s")
    argv = to_argv(plan, "examples/181L.pdb", "runs/x")
    args = build_parser().parse_args(argv)
    assert args.quantity == "backbone" and args.residues == "A:110-112"
    assert args.max_replicas == 8 and args.max_ns == 50 and args.allow == "stop_early,add_replicas"


def test_kt_constant_is_in_kcal():
    assert KB_KCAL * 300 == pytest.approx(0.596, abs=1e-3)


def test_finished_parts_are_read_once_and_cached(tmp_path, monkeypatch):
    import mdtraj as md

    from mdagent.agent import Goal
    from mdagent.tools_openmm import backbone_angles

    run_dir, state, _ = make_run(tmp_path, goal=Goal.for_mode("auto", quantity="backbone_basins"))
    state.backbone = ["rama:A:109-113", "turn:A:110"]
    first, _ = backbone_angles(run_dir, state, 3)
    assert list(first) == ["rama:A:109", "rama:A:110", "rama:A:111", "rama:A:112", "rama:A:113", "turn:A:110"]
    assert not (run_dir / "backbone" / "cache").exists()          # No part is finished: nothing cached.
    for r in range(2):
        for k in range(3):
            (run_dir / f"replica_{r}_part{k:03d}.state.xml").write_text("done")
    backbone_angles(run_dir, state, 3)
    assert len(list((run_dir / "backbone" / "cache").glob("*.npz"))) == 6
    reads = []
    real = md.load_dcd
    monkeypatch.setattr(md, "load_dcd", lambda *a, **k: reads.append(a) or real(*a, **k))
    again, _ = backbone_angles(run_dir, state, 3)
    assert reads == []                                             # All from the cache.
    for label in first:
        for x, y in zip(first[label][1], again[label][1]):
            assert np.allclose(x, y)


def test_patience_is_decided_from_the_budget_unless_given():
    from mdagent.agent import LoopSettings, no_change_hopeless
    from mdagent.diagnose import ResidueReport

    r = ResidueReport("A", 1, "ALA", transitions=[0, 0, 0], occupancies=[], dominant=[])
    loop = LoopSettings(max_ns=1000.0)                   # Needs 4 changes in each of 3 replicas = 12.
    assert not no_change_hopeless(r, 100.0, loop)        # 3 x 900 / 100 = 27 changes still possible.
    assert no_change_hopeless(r, 250.0, loop)            # 3 x 750 / 250 = 9 < 12.
    assert no_change_hopeless(r, 1000.0, loop)           # Budget used up.
    assert not no_change_hopeless(r, 4.0, LoopSettings(max_ns=20.0))
    assert no_change_hopeless(r, 5.0, LoopSettings(max_ns=20.0))
    assert no_change_hopeless(r, 10.0, LoopSettings(max_ns=1000.0, no_flip_patience_ns=10.0))   # Given by the user.
