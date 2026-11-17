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
