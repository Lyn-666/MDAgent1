"""Registries of estimators and tools, and the interfaces they must follow.

Estimator: turns data of one estimand into an Assessment. It serves one estimand kind and is valid for
some data regimes ("unbiased": plain trajectories; "biased": trajectories with reweighting weights;
"seeded": segments started from chosen frames). The registry picks the estimator by (kind, regime), so
switching a method also switches to an estimator that is valid for the new data.

Tool: runs a simulation method. It advances replicas to a length, reports how far it is, and returns
the data of the estimands it was asked for. Its cost factor converts replica-length into budget units.
"""

from __future__ import annotations

from typing import Any, Protocol


class Estimator(Protocol):
    name: str
    kind: str
    regimes: tuple

    def assess(self, estimand: str, data: Any, params: dict): ...   # -> evidence.Assessment


class Tool(Protocol):
    name: str
    regime: str
    cost_factor: float

    def supports(self, kind: str, estimand: str) -> bool: ...   # Can this tool give data for this estimand?
    def length(self) -> float: ...              # Length per replica run so far with this tool.
    def replicas(self) -> int: ...
    def advance(self, n_replicas: int, until: float) -> None: ...  # New replicas first catch up.
    def observe(self, estimands: list) -> dict: ...                # {estimand: data}


class EstimatorRegistry:
    def __init__(self, estimators=()):
        self._by_key: dict[tuple[str, str], Estimator] = {}
        for e in estimators:
            self.add(e)

    def add(self, e: Estimator) -> None:
        for regime in e.regimes:
            key = (e.kind, regime)
            if key in self._by_key:
                raise ValueError(f"two estimators for kind {e.kind!r} and regime {regime!r}")
            self._by_key[key] = e

    def find(self, kind: str, regime: str) -> Estimator | None:
        return self._by_key.get((kind, regime))

    def describe(self) -> list[dict]:
        return [{"kind": k, "regime": r, "estimator": e.name} for (k, r), e in sorted(self._by_key.items())]


class ToolRegistry:
    def __init__(self, tools=()):
        self._tools: dict[str, Tool] = {}
        for t in tools:
            if t.name in self._tools:
                raise ValueError(f"two tools named {t.name!r}")
            self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"no tool named {name!r}; known: {sorted(self._tools)}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)
