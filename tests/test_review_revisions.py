"""Revisions after the review of 2026-11-23: Student's t, automated burn-in, independent trials, harder synthetic
families, the pilot-then-fixed baseline, per-estimand calibration and the sensitivity run. No GPU."""

import numpy as np
import pytest

from mdagent.allocation import allocate
from mdagent.basins import judge, state_factors, trim
from mdagent.experiment_a import (FAMILIES, Design, Targets, family_pool, family_truth, model_pool, run,
                                  sensitivity, sensitivity_markdown)
from mdagent.uncertainty import (Precision, equilibration_start, statistical_inefficiency, t_factor)

SMALL = Design(families=(1, 4), sequential=(4,), max_ns=40.0, budget_ns=320.0, scan_ns=10.0)


# --- Student's t ---------------------------------------------------------------------------------

def test_t_factor_keeps_the_tail_probability():
    assert t_factor(1.959964, 2) == pytest.approx(4.303, abs=1e-3)       # 95 %, 3 replicas
    assert t_factor(1.959964, 1000) == pytest.approx(1.962, abs=1e-2)     # many replicas: back to normal
    assert t_factor(2.0, None) == 2.0 and t_factor(2.0, 0) == 2.0


def test_only_the_confidence_criterion_uses_t():
    conf = Precision(0.05, "confidence", 0.05, looks=3)
    assert conf.factor(2) > conf.factor(None) == pytest.approx(conf.z)
    se = Precision(0.05)                                                 # plain standard error: not an interval
    assert se.factor(2) == 1.0
    # A population with SE 0.02: precise enough with z, not with t on 2 degrees of freedom.
    assert conf.met({"a": 0.02}) and not conf.met({"a": 0.02}, df=2)


def test_the_benchmark_verdict_uses_t_where_the_replica_spread_dominates():
    est = {"se": {"a": 0.02, "b": 0.02}, "transitions": [9, 9, 9], "rates": {}, "dg_se": 0.1,
           "n_replicas": 3, "within_by_state": {"a": 0.01, "b": 0.03}, "between_by_state": {"a": 0.02, "b": 0.01}}
    zs = state_factors(est, 1.96)
    assert zs["a"] == pytest.approx(4.30, abs=0.01) and zs["b"] == pytest.approx(1.96)
    assert not judge(est, 1.96, 0.05)["populations"]                     # 4.30 x 0.02 > 0.05
    assert judge(est, 1.96, 0.05, t_dist=False)["populations"]           # the old rule passed it


def test_a_spread_larger_only_by_chance_keeps_z():
    from mdagent.uncertainty import replica_spread_real

    assert not replica_spread_real(0.02, 0.025, 3)      # ratio 1.6 < chi2(0.95, 2) / 2 = 3.0
    assert replica_spread_real(0.01, 0.025, 3)          # ratio 6.25
    assert not replica_spread_real(0.0, 0.02, 3) and not replica_spread_real(0.01, 0.05, 1)
    est = {"se": {"a": 0.02}, "n_replicas": 3, "within_by_state": {"a": 0.016}, "between_by_state": {"a": 0.02}}
    assert state_factors(est, 2.39)["a"] == 2.39


def test_with_t_more_replicas_also_narrow_the_factor():
    split = [{"within": 0.01, "between": 0.025}]
    plain = allocate(split, 4, 10.0, 2.4, 0.05, 12, 100.0, 0.6)
    with_t = allocate(split, 4, 10.0, 2.4, 0.05, 12, 100.0, 0.6, t_dist=True)
    assert with_t is not None and with_t[0] >= plain[0]                 # t asks for at least as many replicas


def test_live_verdicts_use_t_when_the_error_comes_from_the_replicas():
    from mdagent.diagnose import ResidueReport
    from mdagent.diagnose import judge as live_judge

    def report(within, between):
        return ResidueReport(chain="A", resseq=1, resname="X", transitions=[9, 9, 9],
                             occupancies=[{}] * 3, dominant=["g+"] * 3, ns_per_replica=10.0,
                             populations={"g+": 0.6, "t": 0.4}, standard_error={"g+": 0.018, "t": 0.018},
                             replicas_differ=False, error_from_replicas=between > within,
                             error_within=within, error_between=between)
    conf = Precision(0.05, "confidence", 0.05, looks=3)
    inside, chance, real = report(0.018, 0.010), report(0.014, 0.018), report(0.006, 0.018)
    for r in (inside, chance, real):
        live_judge(r, 4, conf)
    # Only a spread the chi-square test calls real makes the error a 3-number estimate (t, 2 degrees of freedom).
    assert inside.verdict == "ok" and chance.verdict == "ok" and real.verdict == "imprecise"


# --- automated burn-in ---------------------------------------------------------------------------

def _two_state(stay: float, n: int, rng) -> np.ndarray:
    s = [0]
    for _ in range(n - 1):
        s.append(s[-1] if rng.random() < stay else 1 - s[-1])
    return np.array(s)


def test_statistical_inefficiency_of_a_known_chain():
    s = _two_state(0.98, 100_000, np.random.default_rng(1))
    assert statistical_inefficiency(s) == pytest.approx((1 + 0.96) / (1 - 0.96), rel=0.25)   # 49
    assert statistical_inefficiency(np.zeros(100)) == 1.0


def test_burn_in_finds_a_start_transient_and_keeps_an_equilibrium_series():
    rng = np.random.default_rng(0)
    eq = rng.integers(0, 2, 8000)
    biased = np.concatenate([np.ones(2000, int), rng.integers(0, 2, 8000)])
    assert equilibration_start(eq) < 500
    assert 1800 <= equilibration_start(biased) <= 3000
    assert equilibration_start(np.r_[np.full(50, -1), eq]) >= 50          # frames before the first core go
    states, values, starts = trim([biased, eq], [biased * 1.0, eq * 1.0], 2, "auto")
    assert len(states[0]) == len(biased) - starts[0] and len(values[1]) == len(eq) - starts[1]
    assert trim([biased], None, 2, "none")[2] == [0]


def test_burn_in_removes_the_bias_of_a_crystal_start():
    pool = family_pool("biased_start", seed=3, n_replicas=6, ns=40.0)
    from mdagent.experiment_a import estimate
    truth = family_truth("biased_start")["betaI"]
    raw = estimate(pool, "biased_start", list(range(6)), 40.0, Design(burn_in="none"))
    cut = estimate(pool, "biased_start", list(range(6)), 40.0, Design(burn_in="auto"))
    assert abs(cut["populations"]["betaI"] - truth) < abs(raw["populations"]["betaI"] - truth)


def test_new_projection_runs_use_burn_in_and_old_runs_keep_their_setting():
    from argparse import Namespace

    from mdagent.agent import LoopSettings
    from mdagent.cli import burn_in_default

    assert LoopSettings(**{"first_ns": 1.0, "max_ns": 20.0}).burn_in == "none"     # an old run.json
    assert burn_in_default(Namespace(allocation="projection", burn_in=None)) == "auto"
    assert burn_in_default(Namespace(allocation="rounds", burn_in=None)) == "none"
    assert burn_in_default(Namespace(allocation="projection", burn_in="none")) == "none"


def test_backbone_diagnosis_reports_the_burn_in():
    from mdagent.backbone import BackboneSettings, diagnose_backbone
    from mdagent.basins import BasinSet

    pool = family_pool("biased_start", seed=1, n_replicas=3, ns=20.0)
    spec = BasinSet.parse("turn:A:52")
    items = {"turn:A:52": (spec, pool.values["biased_start"])}
    _, details = diagnose_backbone(items, BackboneSettings(), 20.0, pool.ns_per_frame, 300.0, burn_in="auto")
    assert max(details["turn:A:52"]["burn_in_frames"]) > 0
    _, details = diagnose_backbone(items, BackboneSettings(), 20.0, pool.ns_per_frame, 300.0)
    assert details["turn:A:52"]["burn_in_frames"] == [0, 0, 0]


# --- synthetic families --------------------------------------------------------------------------

def test_families_have_their_documented_truth_and_starts():
    assert family_truth("equilibrium")["betaI"] == pytest.approx(0.7)
    assert family_truth("two_timescale")["betaI"] == pytest.approx(0.55)
    assert family_truth("hidden_slow_start") == pytest.approx(family_truth("two_timescale"))
    p = family_pool("hidden_slow_start", seed=0, n_replicas=4, ns=10.0)
    assert all(s[0] == 0 for s in p.states["hidden_slow_start"])        # every replica starts on one side
    q = family_pool("biased_start", seed=0, n_replicas=4, ns=10.0)
    assert all(s[0] == 1 for s in q.states["biased_start"])             # in the minor basin's crystal substate
    long = family_pool("equilibrium", seed=0, n_replicas=4, ns=200.0)
    assert np.mean([np.mean(s == 0) for s in long.states["equilibrium"]]) == pytest.approx(0.7, abs=0.03)
    assert set(FAMILIES) == {"equilibrium", "biased_start", "two_timescale", "hidden_slow_start"}


# --- independent trials, the pilot baseline, per-estimand calibration ----------------------------

def test_fresh_pools_make_every_trial_independent_and_use_the_truth():
    out = run(None, SMALL, Targets(), trials=3, seed=0,
              pool_factory=lambda sd: family_pool("equilibrium", sd, 8, 40.0))
    assert out["independence"] == {"mode": "fresh_pools", "units": 3, "note": out["independence"]["note"]}
    assert out["reference"]["equilibrium"]["populations"]["betaI"] == pytest.approx(0.7)
    a = out["summary"]["agent"]
    assert a["independent_units"] == 3 and "coverage_at_stop" in a and "equilibrium" in a["per_item"]
    assert set(out["arms"]) == {"sequential_ci_4", "pilot_fixed", "agent"}


def test_disjoint_groups_share_no_replica():
    pool = model_pool([([0.7, 0.3], 8.0)], n_replicas=20, ns=40.0, seed=0)
    d = Design(families=(1, 4), sequential=(4,), max_ns=40.0, budget_ns=320.0, scan_ns=10.0, max_replicas=4)
    out = run(pool, d, Targets(), trials=10, holdout=4, seed=0, disjoint=True)
    assert out["independence"]["mode"] == "disjoint_groups" and out["trials"] == 4 == out["independence"]["units"]


def test_a_shared_pool_reports_few_independent_units_and_wide_intervals():
    pool = model_pool([([0.7, 0.3], 8.0)], n_replicas=12, ns=40.0, seed=0)
    out = run(pool, SMALL, Targets(), trials=6, holdout=4, seed=0)
    a = out["summary"]["agent"]
    assert out["independence"]["units"] == 1 == a["independent_units"]
    assert a["success_interval"][1] - a["success_interval"][0] > 0.5    # one unit says little


def test_the_pilot_baseline_projects_once_and_looks_once_more():
    out = run(None, SMALL, Targets(), trials=2, seed=0,
              pool_factory=lambda sd: family_pool("equilibrium", sd, 8, 40.0))
    for t in out["arms"]["pilot_fixed"]:
        assert t["looks"] == 2 and t["steps"][0]["planning"]
        assert t["decision"] in ("STOP", "INSUFFICIENT_EVIDENCE")
    assert "vs_pilot_fixed" in out["verdict"]


def test_sensitivity_runs_one_constant_at_a_time():
    rows = sensitivity(["equilibrium"], SMALL, Targets(), trials=2, constants={"margin": (1.0, 1.2)},
                       n_replicas=8, ns=40.0)
    assert [(r["constant"], r["value"]) for r in rows] == [("margin", 1.0), ("margin", 1.2)]
    assert "(default)" in sensitivity_markdown(rows)
