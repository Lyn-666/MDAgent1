"""Conformational clustering of a finished run (GROMOS method, Daura et al., Angew. Chem. 1999).

The frames after the non-equilibrated start of every replica are pooled and, if there are too many, thinned evenly
to at most `max_frames`. All pairwise RMSDs are computed after superposition on the chosen atoms. GROMOS then
repeats: the frame with the most neighbours within `cutoff_nm` becomes a cluster centre; it and its neighbours form
the cluster and are removed. Every frame belongs to exactly one cluster.

Each cluster gets its population (share of the frames) with an error from the spread between replicas (the share
in each replica, standard error over replicas), the replicas that visit it, and a representative structure (the
centre frame, protein atoms) written as clusters/cluster_<k>.pdb.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def gromos(rmsd_matrix: np.ndarray, cutoff_nm: float) -> tuple[np.ndarray, list[int]]:
    """Cluster label of every frame (0 = largest cluster) and the centre frame of every cluster."""
    n = len(rmsd_matrix)
    neighbours = rmsd_matrix <= cutoff_nm
    left = np.ones(n, dtype=bool)
    labels = np.full(n, -1, dtype=int)
    centres = []
    while left.any():
        counts = (neighbours & left[None, :]).sum(axis=1)
        counts[~left] = -1
        c = int(np.argmax(counts))
        members = neighbours[c] & left
        labels[members] = len(centres)
        centres.append(c)
        left &= ~members
    return labels, centres


def cluster(trajs: list, cutoff_nm: float = 0.15, start_frame: int = 0, max_frames: int = 2500,
            max_clusters: int = 10) -> dict:
    """Cluster the frames from `start_frame` on of each replica (trajectories holding the chosen atoms).

    Returns populations with errors, replicas per cluster, centre (replica, frame) per cluster and the labels."""
    import mdtraj as md

    kept = [t[start_frame:] for t in trajs if t.n_frames > start_frame]
    total = sum(t.n_frames for t in kept)
    if total < 2:
        return {}
    stride = max(1, math.ceil(total / max_frames))
    parts, origin = [], []
    for i, t in enumerate(kept):
        sub = t[::stride]
        parts.append(sub)
        origin += [(i, start_frame + k * stride) for k in range(sub.n_frames)]
    joined = md.join(parts) if len(parts) > 1 else parts[0]
    n = joined.n_frames
    m = np.empty((n, n), dtype=np.float32)
    for k in range(n):
        m[k] = md.rmsd(joined, joined, k)
    m = np.maximum(m, m.T)          # numerical symmetry
    labels, centres = gromos(m, cutoff_nm)
    rep_of = np.array([o[0] for o in origin])
    n_rep = len(kept)
    out = []
    for c, centre in enumerate(centres):
        shares = [float(np.mean(labels[rep_of == r] == c)) if np.any(rep_of == r) else 0.0 for r in range(n_rep)]
        pop = float(np.mean(labels == c))
        se = float(np.std(shares, ddof=1) / math.sqrt(n_rep)) if n_rep > 1 else float("nan")
        out.append({"cluster": c + 1, "population": pop, "error": se, "share_per_replica": shares,
                    "replicas_visiting": [r for r in range(n_rep) if shares[r] > 0],
                    "centre": {"replica": origin[centre][0], "frame": origin[centre][1]},
                    "frames": int(np.sum(labels == c))})
    small = out[max_clusters:]
    return {"cutoff_nm": cutoff_nm, "frames_used": n, "stride": stride, "start_frame": start_frame,
            "n_clusters": len(out), "clusters": out[:max_clusters],
            "rest_population": float(sum(c["population"] for c in small)),
            "shared_by_all_replicas": [c["cluster"] for c in out[:max_clusters]
                                       if len(c["replicas_visiting"]) == n_rep]}


def _frame(run_dir: Path, top, replica: int, frame: int, n_parts: int):
    """One frame (protein atoms) of a replica, read from the part that holds it (not the whole trajectory)."""
    import mdtraj as md
    from mdtraj.formats import DCDTrajectoryFile

    for k in range(n_parts):
        path = run_dir / f"replica_{replica}_part{k:03d}.dcd"
        with DCDTrajectoryFile(str(path)) as f:
            n = len(f)
        if frame < n:
            return md.load_frame(str(path), frame, top=top)
        frame -= n
    raise IndexError(f"frame beyond the end of replica {replica}")


def write(run_dir, top, n_parts: int, result: dict) -> list[Path]:
    """clusters/clusters.json and one PDB (protein atoms) per reported cluster centre."""
    run_dir = Path(run_dir)
    folder = run_dir / "clusters"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in result.get("clusters", []):
        t = _frame(run_dir, top, c["centre"]["replica"], c["centre"]["frame"], n_parts)
        path = folder / f"cluster_{c['cluster']}.pdb"
        t.save_pdb(str(path))
        c["pdb"] = str(path.relative_to(run_dir))
        paths.append(path)
    (folder / "clusters.json").write_text(json.dumps(result, indent=1))
    return [folder / "clusters.json"] + paths


def run(run_dir, keys: list, n_parts: int, cutoff_nm: float = 0.15, start_ns: float = 0.0, dt_ns: float = 0.005,
        min_ca: int = 5) -> dict:
    """Cluster a run's plain MD (all replicas) on the stability atoms of `keys` (empty: the whole protein)."""
    import mdtraj as md

    from .diagnose import load_run
    from .stability import atom_selection

    run_dir = Path(run_dir)
    top = md.load_topology(str(run_dir / "protein.pdb"))
    atoms, _ = atom_selection(top, keys, min_ca)
    trajs = load_run(run_dir, [], n_parts, atoms=atoms)
    result = cluster(trajs, cutoff_nm, start_frame=int(round(start_ns / dt_ns)) if dt_ns else 0)
    if not result:
        return {}
    write(run_dir, top, n_parts, result)   # representative structures, one frame each
    return result
