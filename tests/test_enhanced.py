import math
from pathlib import Path

import numpy as np
import pytest

from mdagent.enhanced import (
    KB_KJ, ChiBiases, EnhancedResult, MetadSettings, coupled_partners, cramers_v, free_energy, judge,
    reweighted_populations,
)

DATA = Path(__file__).resolve().parent / "data"
T = 300.0
GRID = np.linspace(-math.pi, math.pi, 361)


def test_deposit_is_periodic_and_well_tempered():
    s = MetadSettings()
    b = ChiBiases([(0, 1, 2, 3)], s, T)
    b.deposit(np.array([math.pi - 0.01]))     # Near +180°.
    assert b.bias[0, 0] == pytest.approx(b.bias[0, -1])   # -180° and 180° are the same angle.
    assert b.bias[0, 1] > 0.5 * s.height_kj               # The Gaussian wraps to -179°.
    first = b.bias[0].max()
    for _ in range(50):
        b.deposit(np.array([0.0]))
    b2 = ChiBiases([(0, 1, 2, 3)], s, T, b.bias)
    before = b2.bias_at(0, 0.0)
    b2.deposit(np.array([0.0]))
    added = b2.bias_at(0, 0.0) - before
    assert added < first  # Later Gaussians are lower where the bias is already high.


def test_reweighting_recovers_populations_and_couples_residues():
    kt = KB_KJ * T
    rng = np.random.default_rng(0)
    # Residue 0 visits t and g- equally under bias. The bias is 3 kT higher in g-,
    # so after reweighting g- must be e^3 = 20 times more populated than t.
    chi0 = np.where(rng.random(4000) < 0.5, math.pi - 0.1, -math.pi / 3)
    chi1 = np.full(4000, math.pi / 3)       # Residue 1 sits in g+ and has a flat bias.
    chi = np.stack([chi0, chi1], axis=1)
    bias = np.zeros((2, GRID.size))
    bias[0] = np.where(np.degrees(GRID) < 0, 3 * kt, 0.0)
    bias[0, -1] = bias[0, 0]
    p, ess = reweighted_populations(chi, bias, 0, GRID, T)
    assert p["g-"] / p["t"] == pytest.approx(math.e ** 3, rel=0.15)
    assert ess < 4000
    # Residue 1 reweighted together with residue 0's bias: still all g+.
    p1, _ = reweighted_populations(chi, bias, 1, GRID, T, use=[1, 0])
    assert p1["g+"] == pytest.approx(1.0)


def test_summing_many_biases_collapses_effective_samples():
    """The problem seen in the first 181L test: 8 large biases summed leave few useful samples."""
    rng = np.random.default_rng(1)
    n, k = 20000, 8
    chi = rng.choice([math.pi / 3, math.pi - 0.1, -math.pi / 3], size=(n, k))
    bias = np.zeros((k, GRID.size))
    for i in range(k):
        bias[i] = 25.0 * np.cos(3 * GRID) ** 2 + rng.uniform(0, 15, 3).repeat(121)[:GRID.size]
        bias[i, -1] = bias[i, 0]
    _, ess_all = reweighted_populations(chi, bias, 0, GRID, T, use=list(range(k)))
    _, ess_own = reweighted_populations(chi, bias, 0, GRID, T)
    assert ess_all < 0.02 * n
    assert ess_own > 10 * ess_all


def test_cramers_v_and_coupled_partners():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 3, 5000)
    assert cramers_v(a, a) == pytest.approx(1.0)
    assert cramers_v(a, rng.integers(0, 3, 5000)) < 0.05
    centers = np.radians([60.0, 180.0, -60.0])
    follow = np.where(rng.random(5000) < 0.9, a, rng.integers(0, 3, 5000))  # Mostly copies residue 0.
    chi = np.stack([centers[a], centers[follow], centers[rng.integers(0, 3, 5000)]], axis=1)
    assert coupled_partners(chi, 0, 0.2) == [1]
    assert coupled_partners(chi, 2, 0.2) == []


def test_free_energy_from_bias():
    s = MetadSettings(bias_factor=10)
    bias = np.full_like(GRID, 9.0)
    bias[0] = bias[-1] = 0.0
    f = free_energy(bias, s, T)
    assert f.min() == 0 and f.max() == pytest.approx(10.0)


def _result(pops, halves=None, crossings=(10, 10)):
    r = EnhancedResult("A", 111, "VAL", pops, halves or [[p, p] for p in pops], list(crossings))
    judge(r, MetadSettings(), T)
    return r


def test_too_few_effective_samples_is_not_trusted():
    one = {"g+": 0.01, "t": 0.98, "g-": 0.01}
    r = EnhancedResult("A", 111, "VAL", [one, one], [[one, one], [one, one]], [10, 10], [50.0, 5000.0])
    judge(r, MetadSettings(), T)
    assert r.verdict == "not_converged" and "effective samples" in r.reason


def test_judge_verdicts():
    one = {"g+": 0.01, "t": 0.98, "g-": 0.01}
    one_b = {"g+": 0.03, "t": 0.96, "g-": 0.01}   # Minor states below 5%: treated as negligible.
    two = {"g+": 0.02, "t": 0.60, "g-": 0.38}
    two_b = {"g+": 0.02, "t": 0.55, "g-": 0.43}   # Within 2.5 kJ/mol of "two".
    assert _result([one, one_b]).verdict == "single_state"
    assert "kJ/mol" in _result([one, one]).reason
    assert "missed 43% of the population" in _result([two, two_b]).reason
    assert _result([two, two_b]).verdict == "multiple_states"
    assert _result([one, two]).verdict == "not_converged"                        # Replicas disagree.
    assert _result([two, two], halves=[[one, two], [two, two]]).verdict == "not_converged"  # Halves differ.
    assert _result([one, one], crossings=(1, 9)).verdict == "not_converged"      # Too few crossings.


def test_settings_ignore_old_keys():
    s = MetadSettings.from_dict({"length_ns": 20.0, "agreement_tolerance": 0.1})
    assert s.length_ns == 20.0 and s.max_length_ns == 60.0


def test_old_run_json_enables_enhanced_sampling(tmp_path):
    import json
    from mdagent.agent import RunState
    old = {"pdb": "x.pdb", "target_description": "t", "target": [["A", 111]], "n_replicas": 3,
           "base_seed": 1, "md": {}, "loop": {}, "rounds_ns": [1.0], "pending_round_ns": 0.0}
    (tmp_path / "run.json").write_text(json.dumps(old))
    assert RunState.load(tmp_path).enhanced == {"enabled": True}


@pytest.mark.slow
def test_plain_md_then_enhanced_sampling_and_resume(tmp_path):
    """Tiny peptide on CPU: plain MD stops, enhanced sampling starts by itself, then resumes longer."""
    from dataclasses import asdict
    from mdagent.agent import LoopSettings, RunState, run_all
    from mdagent.build import SetupOptions, build_system
    from mdagent.simulate import MDSettings, SystemFiles, write_inputs

    pdb = DATA / "fragment_108-114.pdb"
    opts = SetupOptions(forcefield="amber14", padding_nm=0.8)
    built = build_system(pdb, opts)
    write_inputs(built, tmp_path)
    # min_crossings is impossible to reach, so every result is "not_converged" and the agent
    # must extend the run by itself: 2 ps, then 3 ps (the limit).
    metad = dict(length_ns=0.002, max_length_ns=0.003, extend_ns=0.001, n_replicas=2, pace_ps=0.1,
                 chunk_ns=0.001, height_kj=2.0, min_crossings=10**6)
    state = RunState(
        pdb=str(pdb), target_description="test", target=[("A", 111), ("A", 114)],
        n_replicas=2, base_seed=1,
        md=MDSettings(equilibration_ps=1.0, save_interval_ps=0.5, platform="CPU"),
        loop=LoopSettings(first_ns=0.002, max_ns=0.002),
        setup=asdict(opts),
        enhanced={"enabled": True, "settings": metad},
    )
    state.save(tmp_path)
    logs = []
    decisions, results = run_all(tmp_path, state, SystemFiles(built.topology, built.positions, built.system),
                                 logs.append)
    assert [d.action for d in decisions if d.key == ("A", 111)] == ["enhanced_sampling"]
    assert {r.label for r in results} == {"VAL111", "PHE114"}
    assert all(len(r.populations) == 2 for r in results)
    assert all(abs(sum(r.mean_populations.values()) - 1) < 1e-6 for r in results)
    d = tmp_path / "enhanced"
    assert (d / "results.txt").exists() and (d / "results.json").exists()
    assert np.load(d / "replica_0_chi.npy").shape == (30, 2)  # Extended to 3 ps / 0.1 ps, 2 residues.
    assert any("Extend enhanced sampling from 2 to 3 ns" in line or "from 0.002 to 0.003" in line
               for line in logs)
    assert any("did not converge within the limit" in line for line in logs)
    assert RunState.load(tmp_path).enhanced["settings"]["length_ns"] == 0.003

    # Resume with a higher limit: it continues from the saved bias and state.
    state = RunState.load(tmp_path)
    state.enhanced["settings"]["max_length_ns"] = 0.004
    logs.clear()
    run_all(tmp_path, state, log=logs.append)
    assert np.load(d / "replica_0_chi.npy").shape == (40, 2)
    assert any("enhanced replica 0: 0.004 of 0.004 ns" in line for line in logs)
