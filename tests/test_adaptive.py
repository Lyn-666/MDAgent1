"""Adaptive sampling. AdaptivePy cannot be installed in every test environment, so a
stand-in module writes seeds.csv in the format that the AdaptivePy documentation gives:
seed_id,policy,traj_id,frame_id,cluster_id,global_index."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mdagent.adaptive import read_seeds
from mdagent.diagnose import ResidueReport, judge_pooled

DATA = Path(__file__).resolve().parent / "data"


def _write_seeds(path, rows):
    path.write_text("seed_id,policy,traj_id,frame_id,cluster_id,global_index\n"
                    + "".join(f"{i},least_counts,{t},{f},0,{g}\n" for i, (t, f, g) in enumerate(rows)))


def test_read_seeds_checks_traj_frame_and_global_index(tmp_path):
    csv = tmp_path / "seeds.csv"
    _write_seeds(csv, [(0, 3, 3), (1, 2, 12)])  # Trajectory 0 has 10 frames.
    assert read_seeds(csv, [10, 5]) == [(0, 3), (1, 2)]
    _write_seeds(csv, [(1, 2, 2)])              # global_index does not match.
    with pytest.raises(ValueError, match="does not match"):
        read_seeds(csv, [10, 5])
    _write_seeds(csv, [(1, 7, 17)])             # Frame beyond the end of trajectory 1.
    with pytest.raises(ValueError):
        read_seeds(csv, [10, 5])


def test_judge_pooled_uses_total_flips_only():
    def rep(transitions, dominant):
        r = ResidueReport("A", 1, "VAL", transitions, [{} for _ in transitions], dominant)
        judge_pooled(r, min_total_flips=8)
        return r.verdict
    assert rep([0, 0, 0, 0], ["t", "g-", "t", "t"]) == "no_flip"   # Different states are expected here.
    assert rep([3, 0, 1, 0], ["t", "g-", "t", "t"]) == "rare_flips"
    assert rep([5, 0, 4, 0], ["t", "t", "t", "t"]) == "ok"


def _fake_adaptivepy(calls):
    """Stand-in for adaptivepy.run_adaptive_sampling: pick the last frame of each trajectory."""
    def run_adaptive_sampling(config_path):
        cfg = dict(line.split(": ", 1) for line in Path(config_path).read_text().splitlines()
                   if ": " in line and not line.startswith(" "))
        feats = sorted(Path(cfg["features_dir"]).glob("traj_*.npy"))
        counts = [len(np.load(f)) for f in feats]
        offsets = np.concatenate([[0], np.cumsum(counts)])
        n = int(cfg["n_seeds"])
        rows = [(t % len(feats), counts[t % len(feats)] - 1, offsets[t % len(feats)] + counts[t % len(feats)] - 1)
                for t in range(n)]
        out = Path(cfg["output_dir"]) / "least_counts"
        out.mkdir(parents=True)
        _write_seeds(out / "seeds.csv", rows)
        calls.append((config_path, counts))
    mod = types.ModuleType("adaptivepy")
    mod.run_adaptive_sampling = run_adaptive_sampling
    return mod


@pytest.mark.slow
def test_adaptive_round_starts_from_chosen_frames(tmp_path, monkeypatch):
    from dataclasses import asdict

    from mdagent.agent import LoopSettings, RunState, run_all
    from mdagent.build import SetupOptions, build_system
    from mdagent.simulate import MDSettings, SystemFiles, write_inputs

    calls = []
    monkeypatch.setitem(sys.modules, "adaptivepy", _fake_adaptivepy(calls))
    pdb = DATA / "fragment_108-114.pdb"
    opts = SetupOptions(forcefield="amber14", padding_nm=0.8)
    built = build_system(pdb, opts)
    write_inputs(built, tmp_path)
    state = RunState(
        pdb=str(pdb), target_description="test", target=[("A", 111), ("A", 114)],
        n_replicas=2, base_seed=1,
        md=MDSettings(equilibration_ps=1.0, save_interval_ps=0.5, platform="CPU"),
        loop=LoopSettings(first_ns=0.002, extend_ns=0.002, max_ns=0.004, no_flip_patience_ns=1.0),
        setup=asdict(opts),
        enhanced={"enabled": False},
        sampling="adaptive",
        adaptive={"policy": "least_counts", "n_clusters": 2, "full_interval_ps": 0.5},
    )
    state.save(tmp_path)
    logs = []
    decisions, labels = run_all(tmp_path, state, SystemFiles(built.topology, built.positions, built.system),
                                logs.append)

    assert len(calls) == 1 and calls[0][1] == [4, 4]   # Round 2 used 2 segments x 4 full frames.
    saved = RunState.load(tmp_path)
    assert saved.starts == {"1": [[0, 0, 3], [1, 0, 3]]}
    assert any("replica 0: part 1 starts from frame 3 of replica_0_part000.full.dcd" in line for line in logs)
    assert (tmp_path / "replica_1_part001.full.dcd").exists()
    assert saved.rounds_ns == [0.002, 0.002]
    assert decisions and labels == []  # Enhanced sampling is off in this test.

    # Resume after the end: the stored starts are reused, AdaptivePy is not called again.
    run_all(tmp_path, RunState.load(tmp_path), log=logs.append)
    assert len(calls) == 1
