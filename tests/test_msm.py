"""Populations from short segments that start from chosen frames (adaptive sampling): Markov state model."""

import numpy as np

from mdagent.msm import core_fill, populations, populations_with_errors, reversible_stationary


def _chain(T, start, n, rng):
    s = np.empty(n, dtype=int)
    s[0] = start
    for i in range(1, n):
        s[i] = rng.choice(len(T), p=T[s[i - 1]])
    return s


def test_known_model_from_biased_starts():
    # A reversible 3-state model with populations 0.70, 0.25, 0.05.
    pi = np.array([0.70, 0.25, 0.05])
    k = np.array([[0, 0.02, 0.004], [0.056, 0, 0.01], [0.056, 0.05, 0]])   # pi_i k_ij = pi_j k_ji
    T = k.copy()
    np.fill_diagonal(T, 1 - k.sum(axis=1))
    assert np.allclose(pi @ T, pi, atol=1e-12)
    rng = np.random.default_rng(1)
    # 300 short segments, all started in the rare state 2, as adaptive sampling would do.
    segs = [_chain(T, 2, 40, rng) for _ in range(300)]
    naive = np.bincount(np.concatenate(segs), minlength=3) / sum(len(s) for s in segs)
    assert naive[2] > 0.2                          # Counting time is clearly wrong here (true 0.05).
    pops, errs, left_out = populations_with_errors(segs, ["a", "b", "c"], n_boot=100)
    assert left_out == []
    for name, true in zip("abc", pi):
        assert abs(pops[name] - true) < 3 * errs[name] + 0.01, (name, pops[name], errs[name])
        assert errs[name] > 0


def test_reversible_estimate_of_a_simple_count_matrix():
    c = np.array([[90.0, 10.0], [10.0, 40.0]])
    pi = reversible_stationary(c)
    assert np.allclose(pi, [100 / 150, 50 / 150], atol=1e-6)


def test_frames_between_cores_and_states_not_connected():
    assert core_fill(np.array([-1, 0, -1, -1, 1, -1])).tolist() == [-1, 0, 0, 0, 1, 1]
    # State c is entered once and never left: it is not part of the model, and the result says so.
    segs = [np.array([0, 1, 0, 1, 0, 1, 2, 2, 2])]
    pops, left_out = populations(segs, ["a", "b", "c"])
    assert left_out == ["c"] and pops["c"] == 0.0 and abs(pops["a"] + pops["b"] - 1) < 1e-9


def _rotated(top_traj, key, angles_deg):
    """Frames of the peptide with the chi1 of residue `key` set to the given angles."""
    import mdtraj as md

    t = top_traj
    res = next(r for r in t.topology.residues if (r.chain.chain_id, r.resSeq) == key)
    atom = {a.name: a.index for a in res.atoms}
    n, ca, cb = atom["N"], atom["CA"], atom["CB"]
    moving = [a.index for a in res.atoms if a.name not in ("N", "CA", "C", "O", "H", "HA", "CB", "HB")]
    xyz0 = t.xyz[0]
    current = np.degrees(md.compute_dihedrals(t[:1], [[n, ca, cb, atom["CG1"]]])[0, 0])
    frames = []
    axis = xyz0[cb] - xyz0[ca]
    axis /= np.linalg.norm(axis)
    for target in angles_deg:
        theta = np.radians(target - current)
        k = axis
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        xyz = xyz0.copy()
        xyz[moving] = (xyz0[moving] - xyz0[cb]) @ R.T + xyz0[cb]
        frames.append(xyz)
    return md.Trajectory(np.array(frames), t.topology)


def test_adaptive_diagnosis_gives_markov_model_populations():
    from pathlib import Path

    import mdtraj as md

    from mdagent.diagnose import diagnose_trajectories

    pdb = md.load(str(Path(__file__).parent / "data" / "fragment_108-114.pdb"))
    center = {0: 60.0, 1: 180.0, 2: -60.0}
    pi = np.array([0.70, 0.25, 0.05])
    k = np.array([[0, 0.02, 0.004], [0.056, 0, 0.01], [0.056, 0.05, 0]])
    T = k.copy()
    np.fill_diagonal(T, 1 - k.sum(axis=1))
    rng = np.random.default_rng(2)
    segs = [_rotated(pdb, ("A", 111), [center[x] for x in _chain(T, 2, 40, rng)]) for _ in range(150)]
    rep = diagnose_trajectories(segs, [("A", 111)], 1.0, pooled_total_ns=6.0, n_replicas=3, precision=0.05)[0]
    assert rep.extra["populations_from"].startswith("Markov")
    # g+ is state 0 (true 0.70), g- is state 2 (true 0.05) although every segment started in g-.
    assert abs(rep.populations["g+"] - 0.70) < 3 * rep.standard_error["g+"] + 0.02
    assert abs(rep.populations["g-"] - 0.05) < 3 * rep.standard_error["g-"] + 0.02
