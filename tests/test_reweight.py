import math

import numpy as np

from mdagent.diagnose import STATE_CENTERS_DEG
from mdagent.enhanced import KB_KJ, MetadSettings, deposit
from mdagent.reweight import StopRule, analyze, converged, replay

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


def _result(ns, pops, spread):
    return {"ns_per_replica": ns, "residues": [{"residue": "VAL1", "populations": pops, "replica_spread": spread}]}


def test_stop_only_when_replicas_agree_and_nothing_changes():
    rule = StopRule(tol=0.1, min_ns=2.0)
    a = {"g+": 0.2, "t": 0.7, "g-": 0.1}
    b = {"g+": 0.22, "t": 0.69, "g-": 0.09}
    assert converged(_result(3, b, 0.05), _result(2, a, 0.05), rule)[0]
    assert not converged(_result(3, b, 0.05), None, rule)[0]                          # Needs a previous check.
    assert not converged(_result(3, b, 0.30), _result(2, a, 0.05), rule)[0]           # Replicas disagree.
    assert not converged(_result(3, {"g+": 0.4, "t": 0.5, "g-": 0.1}, 0.05), _result(2, a, 0.05), rule)[0]
    assert not converged(_result(1, b, 0.05), _result(0.5, a, 0.05), rule)[0]         # Shorter than min_ns.
