import math
from pathlib import Path

import numpy as np
import pytest

from mdagent.enhanced import (
    KB_KJ, ChiBiases, EnhancedResult, MetadSettings, free_energy, judge, populations,
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


def test_populations_from_free_energy():
    flat = np.zeros_like(GRID)
    p = populations(flat, GRID, T)
    assert all(v == pytest.approx(1 / 3, abs=0.01) for v in p.values())
    # A free energy with a deep well at 180° (t) only.
    well = 30.0 * (1 - np.exp(-((np.abs(GRID) - math.pi) ** 2) / 0.1))
    p = populations(well, GRID, T)
    assert p["t"] > 0.99


def test_free_energy_from_bias():
    s = MetadSettings(bias_factor=10)
    bias = np.full_like(GRID, 9.0)
    bias[0] = bias[-1] = 0.0
    f = free_energy(bias, s, T)
    assert f.min() == 0 and f.max() == pytest.approx(10.0)


def _result(pops, half=None, crossings=(10, 10)):
    r = EnhancedResult("A", 111, "VAL", pops, half or pops, list(crossings))
    judge(r, MetadSettings(), T)
    return r


def test_judge_verdicts():
    one = {"g+": 0.01, "t": 0.98, "g-": 0.01}
    two = {"g+": 0.02, "t": 0.60, "g-": 0.38}
    assert _result([one, one]).verdict == "single_state"
    assert "kJ/mol" in _result([one, one]).reason
    assert _result([two, two]).verdict == "multiple_states"
    assert _result([one, two]).verdict == "not_converged"                  # Replicas disagree.
    assert _result([two, two], half=[one, one]).verdict == "not_converged"  # Still drifting.
    assert _result([one, one], crossings=(1, 9)).verdict == "not_converged"  # Bias too weak or too short.


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
    metad = dict(length_ns=0.002, n_replicas=2, pace_ps=0.1, chunk_ns=0.001, height_kj=2.0)
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
    assert np.load(d / "replica_0_chi.npy").shape == (20, 2)  # 2 ps / 0.1 ps, 2 residues.

    # Resume with a longer enhanced sampling: it continues from the saved bias and state.
    state = RunState.load(tmp_path)
    state.enhanced["settings"]["length_ns"] = 0.003
    logs.clear()
    run_all(tmp_path, state, log=logs.append)
    assert np.load(d / "replica_0_chi.npy").shape == (30, 2)
    assert any("enhanced replica 0: 0.003 of 0.003 ns" in line for line in logs)
