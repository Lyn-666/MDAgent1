import math

import numpy as np

from mdagent.uncertainty import block_standard_error, combine_replicas, effective_samples


def test_uncorrelated_series_gives_the_textbook_error():
    x = (np.random.default_rng(0).random(20000) < 0.3).astype(float)
    assert abs(block_standard_error(x) - math.sqrt(0.3 * 0.7 / 20000)) < 0.0015


def test_correlated_series_gives_a_much_larger_error():
    # Long stretches in one state (about 500 frames each): only about 40 independent pieces.
    rng = np.random.default_rng(1)
    x = np.repeat((rng.random(40) < 0.3).astype(float), 500)
    naive = math.sqrt(x.mean() * (1 - x.mean()) / len(x))
    assert block_standard_error(x) > 5 * naive


def test_weights_and_effective_samples():
    assert effective_samples(np.ones(100)) == 100
    assert effective_samples(np.array([1.0] + [1e-9] * 99)) < 1.01      # One frame dominates.
    x = np.array([1.0, 0.0] * 500)
    assert block_standard_error(x, np.ones(1000)) == block_standard_error(x)


def test_replicas_are_combined_and_disagreement_is_flagged():
    mean, err, disagree = combine_replicas([0.30, 0.32], [0.02, 0.02])
    assert math.isclose(mean, 0.31) and err >= 0.01 and not disagree
    assert combine_replicas([0.1, 0.9], [0.01, 0.01])[2]                 # Clearly different.
    assert not combine_replicas([0.10, 0.13], [0.0, 0.0])[2]             # Below the 0.05 floor.


def test_error_floor_for_states_seen_rarely_or_never():
    from mdagent.diagnose import populations_with_errors
    from mdagent.uncertainty import floor_from_samples, floor_from_transitions

    # Never seen: the floor reflects how little was seen; it shrinks with more state changes.
    assert floor_from_transitions(0.0, 8, 3) > 0.02
    assert floor_from_transitions(0.0, 800, 3) < 0.002
    # Well sampled two-state case: about 2 p (1 - p) / sqrt(N).
    assert abs(floor_from_transitions(0.5, 100, 0) - 0.05) < 0.002
    assert 0 < floor_from_samples(0.999, 1600) < 0.002

    # Three replicas that never leave state 0: the population of state 1 is 0, but its error is not.
    stuck = [np.zeros(400, dtype=int) for _ in range(3)]
    pops, errs, _, _ = populations_with_errors(stuck, ["a", "b"])
    assert pops["b"] == 0 and errs["b"] > 0.1


def test_compare_counts_tiny_differences_as_the_same(tmp_path):
    import json

    from mdagent import report

    def write(folder, p, e):
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps({"p": p, "e": e}))
        return folder

    a, b = write(tmp_path / "a", 0.999, 0.0), write(tmp_path / "b", 0.998, 0.0)
    fake = {a: [{"label": "VAL26", "populations": {"t": 0.999}, "standard_error": {"t": 0.0}}],
            b: [{"label": "VAL26", "populations": {"t": 0.998}, "standard_error": {"t": 0.0}}]}
    old = report.final_results
    report.final_results = lambda run: fake[run]
    try:
        text, agree = report.compare(a, b)
    finally:
        report.final_results = old
    assert agree, text
