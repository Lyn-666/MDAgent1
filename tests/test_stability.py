"""Structure stability (backbone RMSD and Rg) and conformational clustering, on synthetic data with known answers."""

import json
import math

import numpy as np
import pytest

from mdagent.stability import StabilitySettings, analyze, drift, landscape, native

DT = 0.01   # ns per frame


def ar1(n, mean, sd, phi, rng):
    """Correlated noise around a mean (AR(1) with stationary standard deviation sd)."""
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    e = sd * math.sqrt(1 - phi ** 2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, e)
    return mean + x


def steady(n_rep=3, n=3000, rmsd=0.15, rg=1.20, seed=1):
    rng = np.random.default_rng(seed)
    return ([ar1(n, rmsd, 0.02, 0.9, rng) for _ in range(n_rep)],
            [ar1(n, rg, 0.01, 0.9, rng) for _ in range(n_rep)])


def test_a_steady_structure_is_ok_and_its_mean_is_right():
    r, g = steady()
    # 3 x 30 ns: a departure-rate goal of 50 per µs is reachable (bound about 33 per µs).
    reports, details = analyze(r, g, StabilitySettings(max_departure_rate_per_us=50), 0.05, DT, 30.0)
    by = {x.name: x for x in reports}
    assert by["RMSD"].verdict == "ok" and by["Rg"].verdict == "ok"
    assert abs(by["RMSD"].populations["mean"] - 0.15) < 0.01
    assert abs(by["Rg"].populations["mean"] - 1.20) < 0.01
    assert details["items"]["RMSD"]["drift"]["steady"]
    assert details["native"]["fraction"] > 0.99


def test_a_structure_that_keeps_moving_away_is_drifting():
    rng = np.random.default_rng(2)
    ramp = np.linspace(0.1, 0.6, 3000)
    r = [ramp + ar1(3000, 0.0, 0.02, 0.9, rng) for _ in range(3)]
    g = [np.linspace(1.2, 1.4, 3000) + ar1(3000, 0.0, 0.01, 0.9, rng) for _ in range(3)]
    reports, _ = analyze(r, g, StabilitySettings(), 0.05, DT, 30.0, burn_in="none")
    assert {x.name: x.verdict for x in reports}["RMSD"] == "drifting"
    assert "drifting" in reports[0].reason


def test_the_relaxation_at_the_start_is_left_out():
    rng = np.random.default_rng(3)
    relax = np.linspace(0.02, 0.15, 400)       # the structure relaxes away from the crystal, then stays
    r = [np.concatenate([relax + ar1(400, 0, 0.005, 0.5, rng), ar1(4000, 0.15, 0.02, 0.9, rng)]) for _ in range(3)]
    g = [ar1(4400, 1.2, 0.01, 0.9, rng) for _ in range(3)]
    reports, details = analyze(r, g, StabilitySettings(max_departure_rate_per_us=50), 0.05, DT, 44.0)
    assert details["burn_in_frames"] > 0
    assert abs(reports[0].populations["mean"] - 0.15) < 0.01
    assert reports[0].verdict == "ok"


def test_an_imprecise_mean_is_not_ok():
    rng = np.random.default_rng(4)
    # Replicas sit at clearly different RMSD levels: steady each, but the mean over three is not precise.
    r = [ar1(3000, m, 0.02, 0.9, rng) for m in (0.10, 0.20, 0.30)]
    g = [ar1(3000, 1.2, 0.01, 0.9, rng) for _ in range(3)]
    reports, _ = analyze(r, g, StabilitySettings(), 0.05, DT, 30.0)
    assert reports[0].verdict == "imprecise" and reports[0].replicas_differ


def test_drift_is_an_equivalence_check():
    flat = [np.full(100, 0.2) + 0.001 * np.sin(np.arange(100)) for _ in range(3)]
    assert drift(flat, 1.96, 0.02, 3.0)["steady"]
    assert not drift([v[:3] for v in flat], 1.96, 0.02, 3.0)["steady"]   # too short to judge


def test_native_time_departures_and_the_upper_bound():
    s = StabilitySettings()
    stays = [np.full(1000, 0.1)]
    out = native(stays, s, DT)
    assert out["fraction"] == 1.0 and out["departures_per_replica"] == [0]
    assert out["departure_rate_upper_per_us"] == pytest.approx(2.996 / 10.0 * 1000.0, rel=1e-3)
    assert out["needed_total_ns"] == pytest.approx(2.996 / 10.0 * 1000.0, rel=1e-3)   # goal 10 per µs: ~300 ns
    leaves = [np.concatenate([np.full(500, 0.1), np.full(500, 0.5)])]
    out = native(leaves, s, DT)
    assert out["departures_per_replica"] == [1] and out["fraction"] == pytest.approx(0.5)
    assert out["departure_rate_upper_per_us"] == pytest.approx(4.744 / 10.0 * 1000.0, rel=1e-3)   # one departure
    # Hysteresis: wobbling around the cutoff inside the buffer is not a departure.
    wobble = [0.3 + 0.02 * np.sin(np.arange(1000))]
    assert native(wobble, s, DT)["departures_per_replica"] == [0]


def test_the_free_energy_surface_has_its_minimum_at_zero_where_most_frames_are():
    r, g = steady()
    fes = landscape(r, g, 300.0)
    f = fes["free_energy_kcal"]
    assert np.nanmin(f) == 0.0
    i, j = np.unravel_index(np.nanargmin(f), f.shape)
    assert abs((fes["rmsd_edges_nm"][i] + fes["rmsd_edges_nm"][i + 1]) / 2 - 0.15) < 0.03
    assert np.isfinite(fes["error_kcal"][i, j])


def test_save_writes_the_files(tmp_path):
    from mdagent.stability import save

    r, g = steady(n=500)
    _, details = analyze(r, g, StabilitySettings(), 0.05, DT, 5.0)
    paths = save(tmp_path, r, g, details, DT)
    names = {p.name for p in paths}
    assert {"summary.json", "rmsd_rg.csv", "rmsd_rg.svg", "fes_rmsd_rg.csv", "fes_rmsd_rg.svg"} <= names
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["items"]["RMSD"]["verdict"] in ("ok", "imprecise", "drifting", "rare_flips", "too_short")


# ------------------------------------------------------------------------------------------------------------------
# Clustering

def test_gromos_finds_the_groups_largest_first():
    from mdagent.clustering import gromos

    x = np.array([0.0] * 6 + [1.0] * 3 + [5.0])
    m = np.abs(x[:, None] - x[None, :])
    labels, centres = gromos(m, 0.1)
    assert list(labels) == [0] * 6 + [1] * 3 + [2]
    assert len(centres) == 3 and labels[centres[1]] == 1


def _toy_trajectory(shapes, rng):
    import mdtraj as md

    top = md.Topology()
    chain = top.add_chain()
    for i in range(6):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", md.element.carbon, res)
    xyz = np.array([s + rng.normal(0, 0.01, s.shape) for s in shapes], dtype=np.float32)
    return md.Trajectory(xyz, top)


def test_cluster_populations_replicas_and_centres():
    from mdagent.clustering import cluster

    rng = np.random.default_rng(5)
    line = np.array([[0.38 * i, 0, 0] for i in range(6)], dtype=float)
    bent = line.copy()
    bent[3:, 1] = 1.0                                   # a clearly different shape (RMSD about 0.24 nm)
    a = _toy_trajectory([line] * 30 + [bent] * 10, rng)   # replica 0: 75 % straight
    b = _toy_trajectory([line] * 40, rng)                  # replica 1: always straight
    out = cluster([a, b], cutoff_nm=0.15)
    assert out["n_clusters"] == 2
    big, small = out["clusters"]
    assert big["population"] == pytest.approx(70 / 80) and small["population"] == pytest.approx(10 / 80)
    assert big["replicas_visiting"] == [0, 1] and small["replicas_visiting"] == [0]
    assert out["shared_by_all_replicas"] == [1]
    assert small["centre"]["replica"] == 0 and small["centre"]["frame"] >= 30
    assert small["error"] > 0


def test_cluster_skips_the_start_and_thins_long_runs():
    from mdagent.clustering import cluster

    rng = np.random.default_rng(6)
    line = np.array([[0.38 * i, 0, 0] for i in range(6)], dtype=float)
    t = _toy_trajectory([line] * 100, rng)
    out = cluster([t, t], cutoff_nm=0.15, start_frame=20, max_frames=40)
    assert out["stride"] == 4 and out["frames_used"] == 40 and out["start_frame"] == 20


# ------------------------------------------------------------------------------------------------------------------
# The chat plan, the command line and the report

def test_a_chat_plan_for_stability_becomes_a_valid_run():
    from mdagent.cli import build_parser
    from mdagent.llm import RunPlan, to_argv

    plan = RunPlan(target_kind="stability", stability_tolerance_nm=0.03, rmsd_cutoff_nm=0.25, questions=[],
                   summary="x")
    argv = to_argv(plan, "structures/1YRF.pdb", "runs/v")
    args = build_parser().parse_args(argv)
    assert args.quantity == "stability"
    assert args.stability_tolerance == 0.03 and args.rmsd_cutoff == 0.25


def test_clustering_can_be_asked_for_other_quantities():
    from mdagent.cli import build_parser
    from mdagent.llm import RunPlan, to_argv

    plan = RunPlan(target_kind="residues", residues="A:111", cluster=True, questions=[], summary="x")
    args = build_parser().parse_args(to_argv(plan, "examples/181L.pdb", "runs/x"))
    assert args.cluster and args.quantity != "stability"


def test_the_chinese_report_describes_the_atoms_in_chinese():
    from mdagent.report import _selection_zh

    assert _selection_zh("C-alpha atoms of the whole protein (35 atoms)") == "整个蛋白的Cα 原子（35 个原子）"
    assert "重原子" in _selection_zh("heavy atoms of A:VAL111 (7 atoms; fewer than 5 residues)")
    assert _selection_zh("something else") == "something else"


def test_stable_needs_enough_watched_time_for_the_departure_rate_goal():
    """Precise means alone are not "stable": the 95 % bound of the departure rate must reach its goal (default 10 per
    µs, about 300 ns in all without a departure). For every system, not only the one where this was found."""
    short_r, short_g = steady(n=3000)                     # 3 x 30 ns = 90 ns: bound about 33 per µs
    reports, details = analyze(short_r, short_g, StabilitySettings(), 0.05, DT, 30.0)
    by = {x.name: x for x in reports}
    assert by["RMSD"].verdict == "too_short" and by["Rg"].verdict == "ok"
    assert by["RMSD"].extra["rate_needs_total_ns"] == pytest.approx(299.6, rel=1e-2)
    long_r, long_g = steady(n=11000, seed=7)               # 3 x 110 ns = 330 ns: bound about 9 per µs
    reports, _ = analyze(long_r, long_g, StabilitySettings(), 0.05, DT, 110.0)
    assert all(x.verdict == "ok" for x in reports)


def test_rate_upper_is_the_poisson_bound():
    from mdagent.stability import rate_upper

    assert rate_upper(0) == pytest.approx(2.996, abs=1e-3)
    assert rate_upper(1) == pytest.approx(4.744, abs=1e-3)


def test_a_stability_run_extends_straight_to_the_length_the_rate_goal_needs(tmp_path):
    from dataclasses import asdict

    from mdagent.agent import Decision, Goal, LoopSettings, RunState
    from mdagent.policy import allocation_next
    from mdagent.simulate import MDSettings

    r, g = steady(n=200)                                    # 3 x 2 ns: precise and steady, far too short to be "stable"
    reports, _ = analyze(r, g, StabilitySettings(), 0.05, DT, 2.0, burn_in="none")
    state = RunState(pdb="x", target_description="t", target=[], n_replicas=3, base_seed=1, md=MDSettings(),
                     loop=LoopSettings(first_ns=1.0, max_ns=2000.0), rounds_ns=[1.0, 1.0],
                     goal=asdict(Goal(quantity="structure_stability", allocation="projection")))
    state.notes["allocation"] = {"looks": 1, "tests": 0, "counting": 2}
    state.save(tmp_path)
    decisions = [Decision(x.label, "extend", "") for x in reports]
    more = allocation_next(tmp_path, state, reports, decisions)
    # About 300 ns in all over 3 replicas (plus 5 %), not one more small step.
    assert 2.0 + more == pytest.approx(300 / 3 * 1.05, rel=0.05)


def test_the_end_message_of_a_stability_run_speaks_of_rmsd_rg_and_the_rate(tmp_path):
    """The end message must not use the populations template ("every proportion within 0.05")."""
    import json as _json
    from dataclasses import asdict

    from mdagent.agent import Decision, Goal, LoopSettings, RunState, _post_plain_md_messages
    from mdagent.simulate import MDSettings

    state = RunState(pdb="x", target_description="t", target=[], n_replicas=8, base_seed=1, md=MDSettings(),
                     loop=LoopSettings(first_ns=1.0, max_ns=2000.0), rounds_ns=[1.0, 61.9104, 12.5821],
                     goal=asdict(Goal(quantity="structure_stability", allocation="projection")))
    state.save(tmp_path)
    (tmp_path / "stability").mkdir()
    (tmp_path / "stability" / "summary.json").write_text(_json.dumps(
        {"native": {"departure_rate_upper_per_us": 4.96, "watched_ns": 604.0}}))
    _post_plain_md_messages(tmp_path, state, [Decision("RMSD", "done", ""), Decision("Rg", "done", "")], 0.0)
    msgs = [_json.loads(x) for x in (tmp_path / "messages.jsonl").read_text().splitlines()]
    zh, en = msgs[-1]["zh"], msgs[-1]["en"]
    assert "RMSD 和 Rg 都达到稳态" in zh and "每 µs 5 次" in zh and "比例" not in zh
    assert "8 replicas × 75.5 ns" in en and "departure rate at most 5 per µs" in en


def test_the_cluster_table_lists_only_clusters_of_at_least_one_percent(tmp_path):
    import json as _json

    from mdagent.report import _cluster_section, _ns

    pops = [0.97, 0.012, 0.01, 0.004, 0.002, 0.001, 0.001]
    clusters = [{"cluster": i + 1, "population": p, "error": 0.0, "replicas_visiting": [0], "pdb": f"clusters/c{i}.pdb"}
                for i, p in enumerate(pops)]
    (tmp_path / "clusters").mkdir()
    (tmp_path / "clusters" / "clusters.json").write_text(_json.dumps(
        {"cutoff_nm": 0.15, "frames_used": 100, "stride": 49, "n_clusters": 8, "clusters": clusters,
         "rest_population": 0.0}))
    text = "\n".join(_cluster_section(tmp_path, lambda zh, en: en))
    assert "| 3 |" in text and "| 4 |" not in text
    assert "5 more small clusters (each below 1 %) hold <1%" in text
    assert _ns(75.4925) == "75.5" and _ns(603.94) == "604" and _ns(2000) == "2000"
