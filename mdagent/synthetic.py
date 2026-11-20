"""Synthetic systems for developing and testing the decision loop without MD.

A synthetic estimand is a discrete-time Markov chain over named states with a known transition matrix,
so its true stationary populations are known exactly. A SyntheticTool runs replicas of these chains, and
behaves like a simulation tool of the general loop (core/registry.Tool).

Biased tool: frames come from a second, faster chain (`biased`); each frame gets the exact weight
pi_true / pi_biased of its state, as ideal reweighting would give. This lets tests check that switching
to a biased method gives correct populations, and that degenerate weights are caught.

Determinism: every (replica, estimand) has its own random generator seeded from (seed, replica, index),
and frames are drawn one by one, so the data do not depend on how the run was cut into rounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators import StateData


def stationary(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(np.asarray(P, dtype=float).T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


def two_state(p_stay_a: float, p_stay_b: float) -> np.ndarray:
    return np.array([[p_stay_a, 1 - p_stay_a], [1 - p_stay_b, p_stay_b]])


@dataclass
class SyntheticEstimand:
    kind: str                 # "state_populations" or "event"
    P: np.ndarray             # Transition matrix per frame (rows sum to 1).
    names: list
    start: list | None = None  # Starting state per replica (cycled); default state 0.
    biased: np.ndarray | None = None   # Chain under bias (biased tools only).

    @property
    def truth(self) -> dict:
        return dict(zip(self.names, stationary(self.P).round(6)))


class SyntheticTool:
    def __init__(self, name: str, estimands: dict, regime: str = "unbiased", frames_per_unit: int = 100,
                 cost_factor: float = 1.0, seed: int = 0, overrun: float = 0.0):
        self.name, self.regime, self.cost_factor = name, regime, cost_factor
        self.estimands, self.fpu, self.seed = estimands, frames_per_unit, seed
        self.overrun = overrun    # Tests only: a faulty tool that runs longer than asked.
        self._len = 0.0
        self._traj: list[dict] = []   # Per replica: {estimand: list of states}
        self._rng: list[dict] = []

    def supports(self, kind: str, estimand: str) -> bool:
        return estimand in self.estimands and self.estimands[estimand].kind == kind

    def length(self) -> float:
        return self._len

    def replicas(self) -> int:
        return len(self._traj)

    def _chain(self, e: SyntheticEstimand) -> np.ndarray:
        return e.biased if self.regime == "biased" and e.biased is not None else e.P

    def advance(self, n_replicas: int, until: float) -> None:
        until = until + self.overrun
        while len(self._traj) < n_replicas:
            r = len(self._traj)
            self._traj.append({k: [] for k in self.estimands})
            self._rng.append({k: np.random.default_rng([self.seed, r, i]) for i, k in enumerate(self.estimands)})
        frames = int(round(until * self.fpu))
        for r in range(len(self._traj)):
            for k, e in self.estimands.items():
                s, rng = self._traj[r][k], self._rng[r][k]
                cum = np.cumsum(self._chain(e), axis=1)
                if not s:
                    starts = e.start or [0]
                    s.append(int(starts[r % len(starts)]))
                while len(s) < frames:
                    s.append(int(min(np.searchsorted(cum[s[-1]], rng.random(), side="right"), len(e.names) - 1)))
        self._len = max(self._len, until)

    def observe(self, estimands: list) -> dict:
        out = {}
        for k in estimands:
            e = self.estimands[k]
            states = [np.array(t[k], dtype=int) for t in self._traj]
            weights = None
            if self.regime == "biased" and e.biased is not None:
                ratio = stationary(e.P) / stationary(e.biased)
                weights = [ratio[s] for s in states]
            out[k] = StateData(states, list(e.names), weights)
        return out
