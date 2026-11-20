"""Evidence: what an estimator found for one estimand, in a form the general reasoning can read.

An estimator decides `status` with its own criteria:
- "met": the estimand's success requirement holds and every blocking validity check passed;
- "not_met": data exist, but the requirement does not hold yet;
- "indeterminate": the data cannot tell (for example no state change was ever seen, so a trap
  and a single stable state look the same). Never reported as converged.

`checks` are the estimator's validity checks. A blocking check that fails means the estimate must
not be used as a result, whatever its error.

`signals` use a small shared vocabulary (SIGNALS) so that the reasoner can compare evidence of
different kinds without knowing what the numbers measure. All signals are optional.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATUSES = ("met", "not_met", "indeterminate")

SIGNALS = {
    "events_total": "independent events seen so far (for example state changes) in all replicas",
    "events_per_replica": "the same, per replica (list)",
    "events_needed": "events the estimator needs before it can be trusted (its own criterion), or None",
    "uncertainty_ratio": "current uncertainty / required uncertainty (> 1: not precise enough), or None",
    "uncertainty_scaling": "expected exponent of uncertainty vs amount of data (e.g. -0.5), or None",
    "error_within": "uncertainty part from inside the replicas",
    "error_between": "uncertainty part from the spread between replicas",
    "replicas_disagree": "replicas differ beyond their errors (bool)",
    "projectable": "whether rate extrapolation to the requirement is meaningful (bool)",
    "regime": "data regime the estimate came from: unbiased, biased or seeded",
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    blocking: bool = True
    detail: str = ""


@dataclass
class Assessment:
    estimand: str
    estimator: str
    status: str
    estimate: dict = field(default_factory=dict)       # e.g. {"g+": 0.2, ...}
    uncertainty: dict = field(default_factory=dict)    # Standard errors with the same keys, when they exist.
    checks: list = field(default_factory=list)         # Check, ...
    signals: dict = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        unknown = set(self.signals) - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown signals {sorted(unknown)}; add them to evidence.SIGNALS first")

    @property
    def valid(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def passes(self) -> bool:
        """The core's own guard: "met" counts only together with all blocking checks."""
        return self.status == "met" and self.valid

    def to_dict(self) -> dict:
        return asdict(self)
