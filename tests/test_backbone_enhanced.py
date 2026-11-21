"""Enhanced sampling of backbone phi/psi: separable well-tempered metadynamics on phi and psi, reweighted.

The correctness test uses a toy residue with three basins of known populations and high barriers, sampled
by Monte Carlo under the same bias deposits as the real simulation (enhanced.deposit). Reweighting must
recover the true basin populations, which the raw biased samples do not.
"""

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from mdagent.backbone import assign_basins, parse_basins
from mdagent.enhanced import KB_KJ, MetadSettings, deposit
from mdagent.reweight import StopRule, analyze, assess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shadow import make_run  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
KT = KB_KJ * 300.0
WELLS = {"alpha_R": ((-65.0, -40.0), 0.70), "beta": ((-120.0, 135.0), 0.25), "alpha_L": ((60.0, 45.0), 0.05)}


def energy(phi, psi):
    """Narrow wells at the basin centers with the weights above (kJ/mol); barriers of many kT."""
    p = sum(w * math.exp((math.cos(phi - math.radians(c[0])) + math.cos(psi - math.radians(c[1]))) / 0.05)
            for c, w in WELLS.values())
    return -KT * math.log(p)


def toy_backbone_metad(n_samples, s, seed):
    """Monte Carlo on (phi, psi) under one well-tempered 1D bias on phi and one on psi; columns phi, psi."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-math.pi, math.pi, s.grid_points)
    bias = np.zeros((2, s.grid_points))
    x = [math.radians(-65.0), math.radians(-40.0)]

    def total(a, b):
        return energy(a, b) + float(np.interp(a, grid, bias[0])) + float(np.interp(b, grid, bias[1]))

    e, out = total(*x), []
    for _ in range(n_samples):
        for _ in range(20):
            i = int(rng.integers(2))
            trial = list(x)
            trial[i] = (trial[i] + rng.normal(0, 0.35) + math.pi) % (2 * math.pi) - math.pi
            et = total(*trial)
            if et <= e or rng.random() < math.exp(-(et - e) / KT):
                x, e = trial, et
        out.append(list(x))
        deposit(bias, grid, np.array(x), s, KT)
        e = total(*x)
    return np.array(out)


@pytest.fixture(scope="module")
def toy_runs():
    s = MetadSettings(pace_ps=1.0, height_kj=2.0, grid_points=181)
    return s, [toy_backbone_metad(6000, s, seed=40 + r) for r in range(2)]


def test_reweighting_recovers_basin_populations_and_the_landscape(tmp_path, toy_runs):
    s, runs = toy_runs
    (tmp_path / "enhanced").mkdir()
    for r, data in enumerate(runs):
        np.save(tmp_path / "enhanced" / f"replica_{r}_chi.npy", data)
    result = analyze(tmp_path, ["ALA5"], 2, s, 300.0, StopRule(), kind="backbone")
    res = result["residues"][0]
    assert result["kind"] == "backbone" and result["states"] == ["alpha_R", "beta", "alpha_L", "other"]
    for name, (_, true) in WELLS.items():
        assert abs(res["populations"][name] - true) <= max(4 * res["standard_error"][name], 0.05), (name, res)
    assert min(res["transitions_per_replica"]) > 20                 # The bias pushed it between basins.
    # Without reweighting the populations are wrong: the bias flattens the basins.
    basins = parse_basins(None)
    raw = np.mean(assign_basins(np.degrees(runs[0][:, 0]), np.degrees(runs[0][:, 1]), basins) == 0)
    assert abs(raw - 0.70) > 0.1
    # The reweighted landscape: lowest in alpha_R, about kT ln(0.70 / 0.25) higher in beta.
    fes = result["landscapes"]["ALA5"]
    f = fes["free_energy_kcal"]
    e = fes["edges_deg"]
    c = (e[:-1] + e[1:]) / 2
    phi_c, psi_c = np.meshgrid(c, c, indexing="ij")
    which = assign_basins(phi_c.ravel(), psi_c.ravel(), basins).reshape(f.shape)
    p = fes["probability"]
    assert p[which == 0].sum() / p[which == 1].sum() == pytest.approx(0.70 / 0.25, rel=0.35)
    assert res["fes_max_error_kcal"] is not None


def test_stopping_needs_the_landscape_precision_too(tmp_path, toy_runs):
    s, runs = toy_runs
    (tmp_path / "enhanced").mkdir()
    for r, data in enumerate(runs):
        np.save(tmp_path / "enhanced" / f"replica_{r}_chi.npy", data)
    result = analyze(tmp_path, ["ALA5"], 2, s, 300.0, StopRule(min_ns=1.0), kind="backbone")
    loose = assess(result, None, StopRule(min_ns=1.0), 0.2, fes_error_kcal=50.0)
    strict = assess(result, None, StopRule(min_ns=1.0), 0.2, fes_error_kcal=1e-6)
    assert loose == {"ALA5": "reached"} and strict == {"ALA5": "not_yet"}


def test_phi_and_psi_quads_in_column_order():
    from openmm import app

    from mdagent.enhanced import dihedral_quads

    top = app.PDBFile(str(DATA / "fragment_108-114.pdb")).topology
    quads = dihedral_quads(top, [("A", 110), ("A", 111)], "backbone")
    assert len(quads) == 4
    names = [[a.name for a in top.atoms() if a.index in q] for q in quads]
    assert sorted(names[0]) == sorted(["C", "N", "CA", "C"]) and sorted(names[1]) == sorted(["N", "CA", "C", "N"])
    with pytest.raises(ValueError):
        dihedral_quads(top, [("A", 108)], "backbone")    # The first residue has no phi.


def _backbone_run(tmp_path, toy_runs, requested=True):
    from mdagent.agent import Goal

    s, runs = toy_runs
    run_dir, state, _ = make_run(tmp_path, goal=Goal.for_mode("auto", quantity="backbone_basins", precision=0.1))
    state.target = [("A", 111)]
    state.enhanced = {"enabled": True, "requested": requested,
                      "settings": {**asdict(s), "length_ns": 6.0, "chunk_ns": 2.0, "n_replicas": 2},
                      "stop_rule": {"min_ns": 2.0}}
    state.save(run_dir)
    return run_dir, state, runs


def test_the_agent_switches_a_stuck_backbone_residue_to_phi_psi_metadynamics(tmp_path, monkeypatch, toy_runs):
    import mdagent.agent as agent
    import mdagent.enhanced as enhanced
    from mdagent.agent import Decision, RunState

    run_dir, state, runs = _backbone_run(tmp_path, toy_runs)
    monkeypatch.setattr(agent, "run_loop", lambda *a, **k: [Decision("VAL111", "enhanced_sampling", "stuck", ("A", 111),
                                                                      cause="stalled")])
    kinds = []

    def fake_replica(files, run_dir, r, start, residues, s, md, seed, log, until_ns, kind="chi1"):
        kinds.append(kind)
        d = run_dir / "enhanced"
        d.mkdir(exist_ok=True)
        np.save(d / f"replica_{r}_chi.npy", runs[r][: int(round(until_ns * 1000))])
        (d / f"replica_{r}.json").write_text(json.dumps({"ns_done": until_ns}))
    monkeypatch.setattr(enhanced, "run_replica", fake_replica)
    logs = []
    _, labels = agent.run_all(run_dir, state, files=object(), log=logs.append)
    assert labels == ["VAL111"] and set(kinds) == {"backbone"}
    assert any("metadynamics on phi and psi" in line for line in logs)
    pops = json.loads((run_dir / "enhanced" / "populations.json").read_text())
    assert pops["kind"] == "backbone" and "landscapes" not in pops
    assert (run_dir / "backbone" / "VAL111_fes_reweighted.csv").exists()
    saved = RunState.load(run_dir)
    assert saved.enhanced["outcome"]["reason"] in ("target_reached", "budget_exhausted", "stalled")

    # The new core sees the same: basins and landscape from metadynamics, rates not (biased kinetics).
    from mdagent.tools_openmm import MetadTool

    tool = MetadTool(run_dir, saved)
    assert tool.supports("state_populations", "VAL111 basins") and tool.supports("free_energy_surface", "VAL111 landscape")
    assert not tool.supports("transition_rates", "VAL111 transitions")
    data = tool.observe(["VAL111 basins", "VAL111 landscape"])
    from mdagent.estimators import FreeEnergySurface, WeightedPopulations
    a = WeightedPopulations().assess("VAL111 basins", data["VAL111 basins"], {"max_se": 0.1})
    f = FreeEnergySurface().assess("VAL111 landscape", data["VAL111 landscape"], {"max_error_kcal": 5.0})
    assert abs(a.estimate["alpha_R"] - 0.70) < 0.1 and f.signals["regime"] == "biased"


def test_without_permission_the_switch_is_only_suggested(tmp_path, toy_runs):
    from mdagent.agent import RunState

    run_dir, state, _ = _backbone_run(tmp_path, toy_runs, requested=False)
    assert not state.can_switch()                       # auto default: only stop_early.
    state.goal["permissions"] = ["stop_early", "switch_method"]
    assert RunState(**{**state.__dict__}).can_switch()


@pytest.mark.slow
def test_phi_psi_metadynamics_runs_in_openmm(tmp_path):
    """Tiny peptide on CPU: plain MD, then metadynamics on phi and psi of five residues (ten 1D biases)."""
    from mdagent.agent import Goal, LoopSettings, RunState, run_all
    from mdagent.build import SetupOptions, build_system
    from mdagent.simulate import MDSettings, SystemFiles, write_inputs

    opts = SetupOptions(forcefield="amber14", padding_nm=0.8)
    built = build_system(DATA / "fragment_108-114.pdb", opts)
    write_inputs(built, tmp_path)
    state = RunState(pdb="x", target_description="t", target=[("A", n) for n in range(109, 114)], n_replicas=2,
                     base_seed=1, md=MDSettings(equilibration_ps=1.0, save_interval_ps=0.5, platform="CPU"),
                     loop=LoopSettings(first_ns=0.002, max_ns=0.002), setup=asdict(opts),
                     goal=asdict(Goal.for_mode("auto", quantity="backbone_basins")),
                     enhanced={"enabled": True, "requested": True,
                               "settings": dict(length_ns=0.002, n_replicas=2, pace_ps=0.1, chunk_ns=0.001,
                                                height_kj=2.0, save_interval_ps=0.5)})
    state.save(tmp_path)
    logs = []
    _, labels = run_all(tmp_path, state, SystemFiles(built.topology, built.positions, built.system), logs.append)
    assert len(labels) == 5
    assert np.load(tmp_path / "enhanced" / "replica_0_chi.npy").shape == (20, 10)   # 2 ps / 0.1 ps; phi, psi x 5.
    assert np.load(tmp_path / "enhanced" / "replica_1_bias.npy").shape == (10, 361)
    assert any("Reweighted backbone basin populations" in line for line in logs)
