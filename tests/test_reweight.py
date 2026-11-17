import math

import numpy as np

from mdagent.diagnose import STATE_CENTERS_DEG
from mdagent.enhanced import KB_KJ, MetadSettings, deposit
from mdagent.reweight import StopRule, analyze, assess, replay

KT = KB_KJ * 300.0
WEIGHTS = {"g+": 0.2, "t": 0.7, "g-": 0.1}   # The true populations of the toy side chain.


def energy(x):
    """Three narrow wells at g+, t and g- with the populations above (kJ/mol); barriers of several kT."""
    p = sum(w * np.exp(np.cos(x - math.radians(STATE_CENTERS_DEG[k])) / 0.05) for k, w in WEIGHTS.items())
    return -KT * np.log(p)


def toy_metadynamics(n_samples, s, seed):
    """Monte Carlo on one chi1 angle under a growing well-tempered bias; one sample per deposit."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-math.pi, math.pi, s.grid_points)
    bias = np.zeros((1, s.grid_points))
    x, out = math.radians(180.0), []
    for _ in range(n_samples):
        for _ in range(30):
            y = (x + rng.normal(0, 0.25) + math.pi) % (2 * math.pi) - math.pi
            # The bias adds to the energy and pushes the angle out of wells it has visited.
            du = (energy(y) + float(np.interp(y, grid, bias[0]))) - (energy(x) + float(np.interp(x, grid, bias[0])))
            if du <= 0 or rng.random() < math.exp(-du / KT):
                x = y
        out.append([x])
        deposit(bias, grid, np.array([x]), s, KT)
    return np.array(out)


def test_reweighting_recovers_the_true_populations(tmp_path):
    s = MetadSettings(pace_ps=1.0)
    (tmp_path / "enhanced").mkdir()
    for r in range(2):
        np.save(tmp_path / "enhanced" / f"replica_{r}_chi.npy", toy_metadynamics(6000, s, seed=r))
    result = analyze(tmp_path, ["VAL1"], 2, s, 300.0)
    pops = result["residues"][0]["populations"]
    for k, true in WEIGHTS.items():
        assert abs(pops[k] - true) < 0.06, (k, pops)
    # Without reweighting, the bias makes the states look almost equally populated.
    chi = np.load(tmp_path / "enhanced" / "replica_0_chi.npy")
    raw_t = np.mean(np.abs((np.degrees(chi[:, 0]) - 180 + 180) % 360 - 180) < 60)
    assert raw_t < 0.55


def test_replay_rebuilds_the_saved_bias():
    s = MetadSettings()
    chi = np.random.default_rng(0).uniform(-math.pi, math.pi, (200, 2))
    grid = np.linspace(-math.pi, math.pi, s.grid_points)
    bias = np.zeros((2, s.grid_points))
    for x in chi:
        deposit(bias, grid, x, s, KT)
    v, c = replay(chi, s, KT)
    assert v[0] == 0 and c[0] == 0
    # The bias acting on the last sample is the saved bias minus the last Gaussian pair.
    last = np.zeros((2, s.grid_points))
    for x in chi[:-1]:
        deposit(last, grid, x, s, KT)
    assert math.isclose(v[-1], sum(np.interp(chi[-1, i], grid, last[i]) for i in range(2)), rel_tol=1e-12)


def test_reweighting_recovers_the_true_populations_with_honest_errors(tmp_path):
    s = MetadSettings(pace_ps=1.0)
    (tmp_path / "enhanced").mkdir()
    for r in range(2):
        np.save(tmp_path / "enhanced" / f"replica_{r}_chi.npy", toy_metadynamics(6000, s, seed=10 + r))
    res = analyze(tmp_path, ["VAL1"], 2, s, 300.0)["residues"][0]
    for k, true in WEIGHTS.items():
        assert abs(res["populations"][k] - true) <= max(4 * res["standard_error"][k], 0.03), (k, res)
        assert 0 < res["standard_error"][k] < 0.1
    assert min(res["transitions_per_replica"]) > 50 and not res["replicas_disagree"]


# Two coupled side chains: the state of one changes the populations of the other.
JOINT = np.array([[0.02, 0.15, 0.03],   # rows: chi1 of residue 1 in g+, t, g-
                  [0.25, 0.05, 0.30],   # columns: chi1 of residue 2 in g+, t, g-
                  [0.05, 0.10, 0.05]])
CENTERS = [math.radians(STATE_CENTERS_DEG[k]) for k in ("g+", "t", "g-")]


def coupled_energy(x, y):
    p = sum(JOINT[a, b] * math.exp((math.cos(x - CENTERS[a]) + math.cos(y - CENTERS[b])) / 0.05)
            for a in range(3) for b in range(3))
    return -KT * math.log(p)


def coupled_metadynamics(n_samples, s, seed):
    """Two angles, each with its own 1D bias (as in MDAgent), sampled by Monte Carlo."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-math.pi, math.pi, s.grid_points)
    bias = np.zeros((2, s.grid_points))
    xy, out = [math.pi, math.pi], []

    def total(a, b):
        return coupled_energy(a, b) + float(np.interp(a, grid, bias[0])) + float(np.interp(b, grid, bias[1]))

    e = total(*xy)
    for _ in range(n_samples):
        for _ in range(20):
            i = int(rng.integers(2))
            trial = list(xy)
            trial[i] = (trial[i] + rng.normal(0, 0.3) + math.pi) % (2 * math.pi) - math.pi
            et = total(*trial)
            if et <= e or rng.random() < math.exp(-(et - e) / KT):
                xy, e = trial, et
        out.append(list(xy))
        deposit(bias, grid, np.array(xy), s, KT)
        e = total(*xy)
    return np.array(out)


def test_reweighting_of_coupled_side_chains(tmp_path):
    s = MetadSettings(pace_ps=1.0)
    (tmp_path / "enhanced").mkdir()
    for r in range(2):
        np.save(tmp_path / "enhanced" / f"replica_{r}_chi.npy", coupled_metadynamics(5000, s, seed=20 + r))
    result = analyze(tmp_path, ["VAL1", "LEU2"], 2, s, 300.0)
    true = [JOINT.sum(axis=1), JOINT.sum(axis=0)]   # Exact populations of each residue alone.
    for res, exact in zip(result["residues"], true):
        for k, value in zip(("g+", "t", "g-"), exact):
            assert abs(res["populations"][k] - value) <= max(4 * res["standard_error"][k], 0.04), (res, exact)
    assert not result["weights_degenerate"]


def _result(ns, errs, transitions=(40, 40), disagree=False, degenerate=False):
    pops = {"g+": 0.2, "t": 0.7, "g-": 0.1}
    return {"ns_per_replica": ns, "weights_degenerate": degenerate, "residues": [{
        "residue": "VAL1", "populations": pops, "standard_error": {k: errs for k in pops},
        "max_standard_error": errs, "transitions_per_replica": list(transitions), "replicas_disagree": disagree}]}


def test_verdicts_separate_reached_stalled_unreliable_and_not_yet():
    rule = StopRule(min_ns=2.0)
    assert assess(_result(3, 0.02), None, rule, 0.05) == {"VAL1": "reached"}
    # All replicas trapped in one state: tiny errors, full agreement, but no transitions. Not reached.
    assert assess(_result(3, 0.0, transitions=(0, 0)), None, rule, 0.05) == {"VAL1": "stalled"}
    assert assess(_result(3, 0.02, transitions=(40, 3)), None, rule, 0.05) == {"VAL1": "stalled"}
    assert assess(_result(3, 0.02, degenerate=True), None, rule, 0.05) == {"VAL1": "unreliable"}
    assert assess(_result(3, 0.08), None, rule, 0.05) == {"VAL1": "not_yet"}                 # Too uncertain.
    assert assess(_result(3, 0.02, disagree=True), None, rule, 0.05) == {"VAL1": "not_yet"}
    assert assess(_result(1, 0.02), None, rule, 0.05) == {"VAL1": "not_yet"}                 # Too short.
    # Stability is only reported, never a reason by itself.
    now, before = _result(3, 0.08), _result(2, 0.08)
    assert assess(now, before, rule, 0.05) == {"VAL1": "not_yet"} and now["residues"][0]["change_since_last"] == 0
