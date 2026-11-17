"""Uncertainty of state populations, for plain MD and for reweighted enhanced sampling.

Frames of a trajectory are correlated in time, so the naive standard error is too
small. Here the error comes from block averaging (Flyvbjerg and Petersen, J. Chem.
Phys. 1989, 91, 461): the series is cut into contiguous blocks, the population is
computed in each block, and the spread of the block values gives the error. The
block length doubles from one frame until fewer than `min_blocks` blocks are left;
the largest error is kept, which is on the safe side when the error has not yet
reached its plateau. Weighted frames (reweighted enhanced sampling) use the weighted
ratio estimator in each block.

Replicas: the combined population is the mean over replicas. Its error is the larger
of (a) the errors inside the replicas, combined, and (b) the spread between replicas
divided by sqrt(number of replicas). Replicas disagree when two of them differ by
more than three combined standard errors (and by more than `min_diff`).

These checks can show that sampling is poor; they cannot prove that it is good
(Grossfield et al., LiveCoMS 2018). For example, if all replicas are trapped in the
same state, the errors are small. That is why the callers also count transitions.
"""

from __future__ import annotations

import math

import numpy as np


def block_standard_error(indicator: np.ndarray, weights: np.ndarray | None = None, min_blocks: int = 8) -> float:
    """Standard error of the (weighted) mean of a 0/1 series, allowing for time correlation."""
    x = np.asarray(indicator, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    n = len(x)
    if n < 2 * min_blocks or w.sum() <= 0:
        return float("nan")
    p = float(np.sum(w * x) / np.sum(w))
    worst, size = 0.0, 1
    while n // size >= min_blocks:
        b = n // size
        wb = w[: b * size].reshape(b, size).sum(axis=1)
        xb = (w[: b * size] * x[: b * size]).reshape(b, size).sum(axis=1)
        keep = wb > 0
        pb = xb[keep] / wb[keep]
        frac = wb[keep] / wb[keep].sum()
        k = int(keep.sum())
        if k >= 2:
            var = float(np.sum(frac ** 2 * (pb - p) ** 2)) * k / (k - 1)
            worst = max(worst, math.sqrt(var))
        size *= 2
    return worst


def effective_samples(weights: np.ndarray) -> float:
    """Kish effective sample size of a set of weights. Equal weights give the number of frames."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w * w)) if w.size else 0.0


def combine_replicas(values: list[float], errors: list[float], min_diff: float = 0.05) -> tuple[float, float, bool]:
    """Mean over replicas, its standard error, and whether two replicas clearly disagree."""
    v = np.asarray(values, dtype=float)
    e = np.nan_to_num(np.asarray(errors, dtype=float), nan=0.0)
    mean = float(np.mean(v))
    within = float(math.sqrt(np.sum(e ** 2)) / len(v))
    between = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    disagree = any(abs(v[i] - v[j]) > max(min_diff, 3 * math.sqrt(e[i] ** 2 + e[j] ** 2))
                   for i in range(len(v)) for j in range(i + 1, len(v)))
    return mean, max(within, between), disagree
