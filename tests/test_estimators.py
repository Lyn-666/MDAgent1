"""Uncertainty assessment: the task-specific estimators on synthetic data with known answers."""

import numpy as np

from mdagent.estimators import (FirstEntryEvent, SegmentMSMPopulations, StateData, UnbiasedPopulations,
                                WeightedPopulations, default_registry)
from mdagent.synthetic import SyntheticEstimand, SyntheticTool, stationary, two_state

NAMES = ["a", "b"]


def _run(P, regime="unbiased", biased=None, n=3, length=40.0, seed=0, start=None):
    e = SyntheticEstimand("state_populations", P, NAMES, start=start, biased=biased)
    t = SyntheticTool("t", {"x": e}, regime=regime, seed=seed)
    t.advance(n, length)
    return t.observe(["x"])["x"], e.truth


def test_unbiased_errors_cover_the_truth_at_the_nominal_rate():
    covered = 0
    for seed in range(30):
        data, truth = _run(two_state(0.97, 0.94), seed=seed, length=20.0)
        a = UnbiasedPopulations().assess("x", data, {"max_se": 0.05})
        covered += abs(a.estimate["a"] - truth["a"]) <= 2 * a.uncertainty["a"]
    assert covered >= 25      # About 95 % expected; block errors are on the safe side.


def test_no_state_change_is_indeterminate_never_met_even_with_a_tiny_error():
    data, _ = _run(two_state(1.0, 1.0))
    a = UnbiasedPopulations().assess("x", data, {"max_se": 0.9})
    assert a.status == "indeterminate" and not a.passes
    assert a.signals["events_total"] == 0


def test_replicas_trapped_in_different_states_fail_the_consistency_check():
    data, _ = _run(two_state(1.0, 1.0), start=[0, 1, 0])
    a = UnbiasedPopulations().assess("x", data, {"max_se": 0.5})
    assert a.signals["replicas_disagree"] and not a.passes
    assert not next(c for c in a.checks if c.name == "replicas_consistent").passed


def test_criteria_belong_to_the_estimand_not_to_the_agent():
    data, _ = _run(two_state(0.97, 0.94), length=20.0)
    loose = UnbiasedPopulations().assess("x", data, {"max_se": 0.2})
    strict = UnbiasedPopulations().assess("x", data, {"max_se": 0.001})
    assert loose.passes and strict.status == "not_met" and strict.signals["uncertainty_ratio"] > 1


def test_reweighting_recovers_the_unbiased_populations():
    P = two_state(0.99999, 0.99998)           # Plain MD would hardly see a change.
    data, truth = _run(P, regime="biased", biased=two_state(0.9, 0.85), length=20.0)
    a = WeightedPopulations().assess("x", data, {"max_se": 0.05})
    assert a.passes
    assert abs(a.estimate["a"] - truth["a"]) <= 3 * a.uncertainty["a"]


def test_degenerate_weights_block_the_result():
    data, _ = _run(two_state(0.9, 0.85), regime="biased", biased=two_state(0.9, 0.85), length=5.0)
    data.weights = [np.where(np.arange(len(w)) == 0, 1e9, 1e-9) for w in data.weights]   # One frame dominates.
    a = WeightedPopulations().assess("x", data, {"max_se": 0.5})
    assert not a.passes
    assert not next(c for c in a.checks if c.name == "weights_not_degenerate").passed


def test_markov_model_corrects_segments_started_from_chosen_states():
    P = two_state(0.98, 0.96)
    truth = stationary(P)[0]
    rng = np.random.default_rng(3)
    segments = []
    for k in range(40):                         # Every segment starts in the rare state b on purpose.
        s = [1]
        for _ in range(60):
            s.append(int(rng.random() > P[s[-1], s[-1]]) ^ s[-1])
        segments.append(np.array(s))
    naive = np.mean(np.concatenate(segments) == 0)
    a = SegmentMSMPopulations().assess("x", StateData(segments, NAMES), {"max_se": 0.1})
    assert abs(a.estimate["a"] - truth) < abs(naive - truth)
    assert abs(a.estimate["a"] - truth) <= 3 * a.uncertainty["a"]


def test_event_needs_entry_from_another_state_and_a_hold():
    est = FirstEntryEvent()
    seen = StateData([np.array([0, 0, 1, 1, 1, 0])], NAMES)
    flicker = StateData([np.array([0, 1, 0, 1, 0, 0])], NAMES)
    start_only = StateData([np.array([1, 1, 1, 1])], NAMES)
    assert est.assess("e", seen, {"target": "b", "hold_frames": 3}).passes
    assert est.assess("e", flicker, {"target": "b", "hold_frames": 3}).status == "not_met"
    assert est.assess("e", start_only, {"target": "b", "hold_frames": 1}).status == "indeterminate"


def test_registry_picks_the_estimator_valid_for_the_data_regime():
    reg = default_registry()
    assert reg.find("state_populations", "biased").name == WeightedPopulations.name
    assert reg.find("state_populations", "seeded").name == SegmentMSMPopulations.name
    assert reg.find("event", "biased") is None     # A bias can force any state: seeing it there proves nothing.
