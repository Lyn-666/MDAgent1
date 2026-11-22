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

Stopping as soon as a result looks precise enough is a sequential decision: the
agent looks at the same growing data set many times, so the plain 95% interval is
too narrow at the moment it decides to stop (the "peeking" problem). `Precision`
holds the tolerance the user asked for and says how wide the interval has to be:

- "standard_error": the plain standard error must be at most the tolerance. This
  is the oldest behaviour of MDAgent and stays the default, so that runs made
  before this existed mean the same thing.
- "confidence": a (1 - alpha) interval, with the alpha divided over the number of
  decision points that are planned in advance (Bonferroni). The number of looks is
  known in advance here, because it is the budget divided by the length of a round.
- "sequence": a confidence sequence (Robbins, Ann. Math. Statist. 1970, normal
  mixture). It is valid however many times the agent looks, also beyond the planned
  budget, and is the widest of the three.
Error floor: block averaging gives an error of exactly 0 for a state that was never
seen, and a too small error for a state seen in a few long visits. A population cannot
be known better than the number of independent visits allows, so the error is never
smaller than a floor:
- plain MD (unbiased): for a two-state jump process the standard error of the time
  fraction p after N state changes is about 2 p (1 - p) / sqrt(N). N is the number of
  state changes in all replicas plus the number of replicas (each replica starts one
  visit).
- reweighted enhanced sampling: the binomial error sqrt(p (1 - p) / n) with n the
  Kish effective number of samples of all replicas.
In both, p is shrunk by half an observation, p' = (p n + 0.5) / (n + 1), so a state
never seen gets an error that reflects how little was seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

CRITERIA = ("standard_error", "confidence", "sequence")


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


def error_parts(values: list[float], errors: list[float]) -> tuple[float, float]:
    """The two parts of the error of a mean over replicas: inside the replicas, and between them.

    When the part between replicas is larger, more independent replicas reduce the error faster
    than longer replicas (agent policy: add replicas).
    """
    v = np.asarray(values, dtype=float)
    e = np.nan_to_num(np.asarray(errors, dtype=float), nan=0.0)
    within = float(math.sqrt(np.sum(e ** 2)) / len(v))
    between = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return within, between


def combine_replicas(values: list[float], errors: list[float], min_diff: float = 0.05) -> tuple[float, float, bool]:
    """Mean over replicas, its standard error, and whether two replicas clearly disagree."""
    v = np.asarray(values, dtype=float)
    e = np.nan_to_num(np.asarray(errors, dtype=float), nan=0.0)
    mean = float(np.mean(v))
    within, between = error_parts(values, errors)
    disagree = any(abs(v[i] - v[j]) > max(min_diff, 3 * math.sqrt(e[i] ** 2 + e[j] ** 2))
                   for i in range(len(v)) for j in range(i + 1, len(v)))
    return mean, max(within, between), disagree


def mixture_factor(n_eff: float, n_target: float, alpha: float) -> float:
    """How many standard errors wide a confidence sequence is (Robbins normal mixture).

    The boundary is tightest at `n_target` effective samples and stays valid however
    many times it is read. n_eff and n_target are numbers of independent samples.
    """
    rho = 1.0 / max(n_target, 1.0)
    u = max(n_eff, 1.0) * rho
    return math.sqrt(2 * (u + 1) / u * math.log(math.sqrt(u + 1) / max(alpha, 1e-12)))


@dataclass
class Precision:
    """The precision the user asked for, and how wide an interval has to be to meet it.

    epsilon: the tolerance on a state population (0 to 1).
    criterion: "standard_error", "confidence" or "sequence" (see the module docstring).
    alpha: the allowed probability that the true value is outside the interval.
    looks: how many decision points are planned, for the "confidence" criterion.
    """

    epsilon: float = 0.05
    criterion: str = "standard_error"
    alpha: float = 0.05
    looks: int = 1

    def __post_init__(self) -> None:
        # A typo must not be read as one of the other criteria by accident.
        if self.criterion not in CRITERIA:
            raise ValueError(f"unknown precision criterion {self.criterion!r}. Use one of {', '.join(CRITERIA)}.")

    @classmethod
    def of(cls, value, **kw) -> "Precision | None":
        """Accept a Precision, a plain number (standard error) or None."""
        if value is None or isinstance(value, cls):
            return value
        return cls(epsilon=float(value), **kw)

    @property
    def z(self) -> float:
        """How many standard errors wide the interval is.

        For "sequence" the width depends on how much data there is; this is its value at the
        point where the boundary is tightest, so it is the smallest factor that criterion uses.
        """
        if self.criterion == "standard_error":
            return 1.0
        if self.criterion == "sequence":
            n = 0.25 / max(self.epsilon, 1e-6) ** 2
            return mixture_factor(n, n, self.alpha)
        return NormalDist().inv_cdf(1 - self.alpha / (2 * max(1, self.looks)))

    def halfwidth(self, standard_error: float, population: float | None = None) -> float:
        """Half-width of the interval of one population."""
        se = float(standard_error)
        if not math.isfinite(se) or se <= 0:
            return se
        if self.criterion != "sequence":
            return self.z * se
        p = 0.5 if population is None else min(max(float(population), 0.0), 1.0)
        n_eff = max(p * (1 - p), 1e-6) / se ** 2
        return mixture_factor(n_eff, 0.25 / max(self.epsilon, 1e-6) ** 2, self.alpha) * se

    def halfwidths(self, errors: dict, populations: dict | None = None) -> dict:
        return {k: self.halfwidth(v, (populations or {}).get(k)) for k, v in errors.items()}

    def met(self, errors: dict, populations: dict | None = None) -> bool:
        """Is every population of this item precise enough?"""
        hw = self.halfwidths(errors, populations).values()
        return bool(hw) and all(math.isfinite(h) and h <= self.epsilon for h in hw)

    def worst(self, errors: dict, populations: dict | None = None) -> float:
        hw = list(self.halfwidths(errors, populations).values())
        return max(hw) if hw else float("nan")

    def describe(self, zh: bool = False) -> str:
        if self.criterion == "standard_error":
            return (f"每个比例的标准误差不超过 {self.epsilon:g}" if zh else
                    f"standard error of every population at most {self.epsilon:g}")
        kind = ("按 {:.0%} 置信区间（{} 次判断，Bonferroni）" if zh else
                "half-width of the {:.0%} interval ({} planned looks, Bonferroni)") if self.criterion == "confidence" \
            else ("按 {:.0%} 置信序列（可以任意多次判断）" if zh else
                  "half-width of the {:.0%} confidence sequence (any number of looks)")
        text = kind.format(1 - self.alpha, self.looks)
        return (f"每个比例的误差不超过 {self.epsilon:g}，{text}" if zh else
                f"{text} at most {self.epsilon:g} for every population")


def planned_looks(first_ns: float, extend_ns: float, max_ns: float, extend_factor: float = 1.0) -> int:
    """How many times the agent will judge the data, if it runs to the end of its budget."""
    cap = max(max_ns * max(extend_factor, 1.0), first_ns)
    step = max(extend_ns, 1e-9)
    return max(1, int(math.ceil((cap - first_ns) / step)) + 1)
def _shrunk(p: float, n: float) -> float:
    return (p * n + 0.5) / (n + 1)


def floor_from_transitions(p: float, transitions: int, n_replicas: int) -> float:
    """Smallest credible standard error of a population from unbiased trajectories (see the module notes)."""
    n = max(transitions + n_replicas, 1)
    q = _shrunk(p, n)
    return 2 * q * (1 - q) / math.sqrt(n)


def floor_from_samples(p: float, n_effective: float) -> float:
    """Smallest credible standard error of a population from n_effective independent (weighted) samples."""
    n = max(n_effective, 1.0)
    q = _shrunk(p, n)
    return math.sqrt(q * (1 - q) / n)


def changes_needed(n_replicas: int, min_changes_per_replica: int = 4, tolerance: float | None = None,
                   z: float = 1.0) -> float:
    """State changes an item needs before its result can be trusted and precise enough.

    Two needs, the larger counts: every replica must change state min_changes_per_replica times (adequate
    sampling), and, with a tolerance, the error floor of even a small basin (population about the tolerance) must
    fall to the tolerance: 2 q (1 - q) / sqrt(N) * z <= tolerance gives N >= (2 (1 - tolerance) z)^2.
    """
    need = float(min_changes_per_replica * n_replicas)
    if tolerance:
        need = max(need, (2 * (1 - tolerance) * z) ** 2)
    return need


def never_changing_hopeless(exposure_ns: float, remaining_ns: float, needed: float) -> bool:
    """Give up on more of the same sampling for an item that has not changed state at all?

    With zero changes in `exposure_ns` of data (all replicas together), the rate is at most 3 / exposure_ns at 95 %
    confidence (Poisson, "rule of three"). Even at that fastest rate, the rest of the budget (`remaining_ns`, all
    replicas together) is expected to give 3 * remaining / exposure changes; when that is fewer than `needed`,
    more of the same is not expected to answer within the limits. No fixed waiting time: more budget means more
    patience. The general decision core uses the same rule (core/reasoner.py).
    """
    if exposure_ns <= 0:
        return False
    return 3.0 * max(remaining_ns, 0.0) / exposure_ns < needed
