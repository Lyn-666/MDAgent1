"""What the user asked for: the estimands, their success requirements, the budget and the permissions.

A Requirement names an estimand (one thing to measure, for example the states of one angle), its kind (for example
"state_populations" or "event") and the success criteria for that kind (`params`, for example
{"max_se": 0.05} or {"seen_in": 1}). The criteria belong to the estimand: different estimands of one
task can have different criteria, and the core never compares a number with a threshold itself.

Budget: the user's limits. `planned_*` is what the user asked for; `hard_*` is the most the agent may
use when the user granted "extend" or "add_replicas". By default hard == planned, so the user's numbers
are limits (user decision 2026-11-20). Cost is counted in replica-length units (for MD: replica-ns),
times the cost factor of the tool.

Modes:
- "guided": the user chose the method and the plan. The agent follows it, never switches the method,
  and only suggests (default permissions: none).
- "auto": the agent may act within the granted permissions (default: only stop_early).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PERMISSIONS = ("extend", "add_replicas", "switch_method", "stop_early")
DEFAULT_PERMISSIONS = {"auto": ("stop_early",), "guided": ()}


@dataclass(frozen=True)
class Requirement:
    estimand: str              # Identifier, unique in the task.
    kind: str                  # Estimand kind; the registry picks an estimator for (kind, data regime).
    params: dict = field(default_factory=dict)  # Success criteria and estimator options for this estimand.
    required: bool = True      # False: reported, but not needed for "target reached".
    confirm_looks: int = 1     # The requirement must hold at this many consecutive looks (repeated-look guard).


@dataclass(frozen=True)
class Budget:
    planned_cost: float        # Replica-length units the user planned (for MD: ns x replicas).
    planned_replicas: int
    hard_cost: float | None = None      # With "extend": the most the agent may spend. Default: planned.
    hard_replicas: int | None = None    # With "add_replicas": the most replicas. Default: planned.
    unit: str = "ns"

    def __post_init__(self):
        if self.planned_cost <= 0 or self.planned_replicas < 1:
            raise ValueError("the budget needs a positive cost and at least one replica")
        if self.max_cost < self.planned_cost or self.max_replicas < self.planned_replicas:
            raise ValueError("hard limits cannot be below the planned budget")

    @property
    def max_cost(self) -> float:
        return self.planned_cost if self.hard_cost is None else self.hard_cost

    @property
    def max_replicas(self) -> int:
        return self.planned_replicas if self.hard_replicas is None else self.hard_replicas


@dataclass(frozen=True)
class TaskSpec:
    question: str                       # The user's words.
    requirements: tuple                 # Requirement, ...
    budget: Budget
    method: str                         # The tool to start with.
    mode: str = "auto"
    permissions: tuple | None = None    # None: the defaults of the mode.
    allowed_methods: tuple = ()         # Tools the agent may switch to (auto mode, with switch_method).
    step: float = 1.0                   # Length per replica of one round (in budget units).

    def __post_init__(self):
        if self.mode not in DEFAULT_PERMISSIONS:
            raise ValueError(f"unknown mode {self.mode!r}")
        bad = set(self.permissions or ()) - set(PERMISSIONS)
        if bad:
            raise ValueError(f"unknown permissions: {sorted(bad)}")
        names = [r.estimand for r in self.requirements]
        if len(names) != len(set(names)) or not names:
            raise ValueError("estimands must be given and unique")

    def granted(self) -> frozenset:
        """Permissions in force. Guided mode never switches the method, whatever was granted."""
        p = set(DEFAULT_PERMISSIONS[self.mode] if self.permissions is None else self.permissions)
        if self.mode == "guided":
            p.discard("switch_method")
        return frozenset(p)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["granted"] = sorted(self.granted())
        return d
