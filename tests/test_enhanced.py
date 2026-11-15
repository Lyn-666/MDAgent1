import math
from pathlib import Path

import numpy as np
import pytest

from mdagent.enhanced import ChiBiases, MetadSettings

DATA = Path(__file__).resolve().parent / "data"
T = 300.0


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


def test_settings_ignore_old_keys():
    s = MetadSettings.from_dict({"length_ns": 20.0, "max_length_ns": 60.0, "agreement_kj": 2.5})
    assert s.length_ns == 20.0 and s.n_replicas == 2


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

    import mdtraj as md

    from mdagent.agent import LoopSettings, RunState, run_all
    from mdagent.build import SetupOptions, build_system
    from mdagent.simulate import MDSettings, SystemFiles, write_inputs

    pdb = DATA / "fragment_108-114.pdb"
    opts = SetupOptions(forcefield="amber14", padding_nm=0.8)
    built = build_system(pdb, opts)
    write_inputs(built, tmp_path)
    metad = dict(length_ns=0.002, n_replicas=2, pace_ps=0.1, chunk_ns=0.001, height_kj=2.0,
                 save_interval_ps=0.5)
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
    decisions, labels = run_all(tmp_path, state, SystemFiles(built.topology, built.positions, built.system),
                                logs.append)
    assert [d.action for d in decisions if d.key == ("A", 111)] == ["enhanced_sampling"]
    assert set(labels) == {"VAL111", "PHE114"}
    d = tmp_path / "enhanced"
    assert np.load(d / "replica_0_chi.npy").shape == (20, 2)  # 2 ps / 0.1 ps, 2 residues.
    assert np.load(d / "replica_1_bias.npy").shape == (2, 361)
    traj = md.load_dcd(str(d / "replica_0_chunk001.dcd"), top=str(tmp_path / "protein.pdb"))
    assert traj.n_frames == 2  # 1 ps chunk / 0.5 ps per frame.
    assert not (d / "results.txt").exists()  # The agent does not analyze.
    assert any("Enhanced sampling finished" in line for line in logs)

    # Resume with a longer enhanced sampling: it continues from the saved bias and state.
    state = RunState.load(tmp_path)
    state.enhanced["settings"]["length_ns"] = 0.003
    logs.clear()
    run_all(tmp_path, state, log=logs.append)
    assert np.load(d / "replica_0_chi.npy").shape == (30, 2)
    assert (d / "replica_1_chunk002.dcd").exists()
    assert any("enhanced replica 0: 0.003 of 0.003 ns" in line for line in logs)
