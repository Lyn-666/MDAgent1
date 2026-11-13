"""Adaptive sampling with AdaptivePy (Shukla group): choose where the next round starts.

In the default mode, each replica continues from its own last state. In
adaptive mode, from round 2 on, each replica starts from a frame chosen by
AdaptivePy, for example a frame in a rarely visited region (least counts).

Features: sin and cos of chi1 for each target residue. Using sin and cos
keeps -180° and 180° close together.

Starting from any frame needs all atoms, including water. So in adaptive
mode every part also saves the full system every full_interval_ps
(replica_<i>_part<k>.full.dcd). Seeds are chosen only from these frames. A
new part starts from the chosen frame with new random velocities.

AdaptivePy is an optional dependency: pip install adaptivepy-sampling
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mdtraj as md
import numpy as np

from .diagnose import _quiet_c_output
from .simulate import part_path

POLICIES = ("least_counts", "random")
INSTALL = "pip install adaptivepy-sampling"


@dataclass
class AdaptiveSettings:
    policy: str = "least_counts"
    n_clusters: int = 20
    full_interval_ps: float = 10.0

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveSettings":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


def check_installed() -> None:
    try:
        import adaptivepy  # noqa: F401
    except ImportError as e:
        raise ValueError(f"Adaptive sampling needs AdaptivePy. Install it with: {INSTALL}") from e


def features(run_dir: Path, segments: list[tuple[int, int]], target: list[tuple[str, int]]) -> list[np.ndarray]:
    """sin/cos of chi1 of the target residues, for every full-system frame of every segment."""
    top = md.load_topology(str(run_dir / "solvated.pdb"))
    wanted = set(target)
    atoms = [a.index for a in top.atoms if (a.residue.chain.chain_id, a.residue.resSeq) in wanted]
    out = []
    with _quiet_c_output():
        for replica, part in segments:
            traj = md.load_dcd(str(part_path(run_dir, replica, part, ".full.dcd")), top=top, atom_indices=atoms)
            _, chi = md.compute_chi1(traj)
            out.append(np.hstack([np.sin(chi), np.cos(chi)]))
    return out


def _write_config(path: Path, features_dir: Path, output_dir: Path, s: AdaptiveSettings,
                  n_clusters: int, n_seeds: int, random_seed: int) -> None:
    path.write_text(
        f"features_dir: {features_dir}\n"
        f"output_dir: {output_dir}\n"
        "clustering:\n"
        "  method: kmeans\n"
        f"  n_clusters: {n_clusters}\n"
        "  params: {}\n"
        "policies:\n"
        f"  - {s.policy}\n"
        f"n_seeds: {n_seeds}\n"
        "seed_selection:\n"
        "  method: nearest_center\n"
        f"random_seed: {random_seed}\n"
        "write_pdbs: false\n"
    )


def read_seeds(seeds_csv: Path, frame_counts: list[int]) -> list[tuple[int, int]]:
    """Return (trajectory index, frame index) for each seed, after checking them.

    AdaptivePy gives both (traj_id, frame_id) and global_index, the row in all
    features stacked in file order. They must agree; otherwise the files were
    read in another order than we wrote them, and we stop instead of starting
    from a wrong frame.
    """
    offsets = np.concatenate([[0], np.cumsum(frame_counts)])
    seeds = []
    with open(seeds_csv) as f:
        for row in csv.DictReader(f):
            t, fr, g = int(row["traj_id"]), int(row["frame_id"]), int(row["global_index"])
            if not (0 <= t < len(frame_counts) and 0 <= fr < frame_counts[t] and offsets[t] + fr == g):
                raise ValueError(f"AdaptivePy seed {dict(row)} does not match the feature files. "
                                 "Stopping instead of starting from a wrong frame.")
            seeds.append((t, fr))
    if not seeds:
        raise ValueError(f"AdaptivePy returned no seeds in {seeds_csv}.")
    return seeds


def choose_starts(
    run_dir: Path,
    round_index: int,
    segments: list[tuple[int, int]],
    target: list[tuple[str, int]],
    n_starts: int,
    s: AdaptiveSettings,
    random_seed: int,
    log: Callable[[str], None] = print,
) -> list[tuple[int, int, int]]:
    """Ask AdaptivePy for n_starts frames. Returns (replica, part, frame) for each."""
    check_installed()
    from adaptivepy import run_adaptive_sampling

    d = run_dir / "adaptive" / f"round_{round_index + 1:03d}"
    feat_dir, out_dir = d / "features", d / "results"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feats = features(run_dir, segments, target)
    for i, f in enumerate(feats):
        np.save(feat_dir / f"traj_{i:03d}.npy", f)  # Zero-padded: file order equals list order.
    counts = [len(f) for f in feats]
    n_clusters = max(2, min(s.n_clusters, sum(counts) // 2))
    config = d / "config.yaml"
    _write_config(config, feat_dir, out_dir, s, n_clusters, n_starts, random_seed)
    run_adaptive_sampling(str(config))

    seeds = read_seeds(out_dir / s.policy / "seeds.csv", counts)
    starts = [(*segments[t], fr) for t, fr in seeds]
    while len(starts) < n_starts:  # Fewer seeds than replicas: reuse them in turn.
        starts.append(starts[len(starts) % len(seeds)])
    starts = starts[:n_starts]
    log(f"AdaptivePy ({s.policy}, {n_clusters} clusters) chose the starting frames: "
        + ", ".join(f"replica {r} part {p} frame {fr}" for r, p, fr in starts) + ".")
    return starts
