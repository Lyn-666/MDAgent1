"""What a run costs, so that "the same result for less computing" can be checked.

Three numbers are kept apart, because they do not grow together:

- aggregate_ns: the production time, summed over all replicas. This is what papers
  usually report as "total simulation time".
- compute_ns: aggregate_ns plus the overhead. Every replica is minimized, heated and
  equilibrated before it produces anything, so ten short replicas cost more than one
  long replica of the same aggregate length. compute_ns is the main cost: it is what
  the GPU actually has to integrate.
- wall_ns: how long the user waits, in simulated ns of the longest chain. With
  n_parallel GPUs the replicas are spread over them (longest first), so more replicas
  can mean a shorter wait at the same compute cost.

A replica that is added late first has to catch up to the length of the others, so its
production time counts in full; the caller passes the length of every replica.

ns are converted to GPU-hours with a measured throughput (ns per day), which depends on
the computer and the system size. A run also records the measured wall-clock seconds of
every round (agent.run_loop), so the model can be checked against the clock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class CostModel:
    """How simulated time is turned into cost. The defaults match MDSettings."""

    equilibration_ns: float = 0.6   # Paid once per replica (600 ps by default).
    analysis_ns: float = 0.0        # Cost of one diagnosis, in ns-equivalents (usually small).
    n_parallel: int = 1             # How many replicas run at the same time.
    ns_per_day: float = 0.0         # Measured throughput, for GPU-hours. 0: unknown.

    @classmethod
    def from_run(cls, state, n_parallel: int = 1, ns_per_day: float = 0.0) -> "CostModel":
        md = state.get("md", {}) if isinstance(state, dict) else {"equilibration_ps": state.md.equilibration_ps}
        return cls(equilibration_ns=md.get("equilibration_ps", 600.0) / 1000.0,
                   n_parallel=n_parallel, ns_per_day=ns_per_day)


@dataclass
class Cost:
    aggregate_ns: float      # Production ns, summed over replicas.
    n_replicas: int
    overhead_ns: float       # Equilibration of every replica, plus the diagnoses.
    compute_ns: float        # aggregate_ns + overhead_ns. The main cost metric.
    wall_ns: float           # Longest chain, after spreading the replicas over n_parallel GPUs.
    looks: int = 0           # How many times the data was judged.
    measured_seconds: float = 0.0  # Real wall-clock, when the run recorded it.

    def gpu_hours(self, ns_per_day: float) -> float:
        return self.compute_ns / ns_per_day * 24.0 if ns_per_day > 0 else float("nan")

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _makespan(chains: list[float], n_parallel: int) -> float:
    """Longest-first scheduling of independent chains on n_parallel GPUs."""
    lanes = [0.0] * max(1, int(n_parallel))
    for c in sorted(chains, reverse=True):
        i = min(range(len(lanes)), key=lanes.__getitem__)
        lanes[i] += c
    return max(lanes) if lanes else 0.0


def of(per_replica_ns: list[float], model: CostModel | None = None, looks: int = 0,
       measured_seconds: float = 0.0) -> Cost:
    """Cost of a run in which replica i produced per_replica_ns[i] ns."""
    m = model or CostModel()
    lengths = [float(x) for x in per_replica_ns if x > 0]
    aggregate = sum(lengths)
    overhead = len(lengths) * m.equilibration_ns + looks * m.analysis_ns
    chains = [m.equilibration_ns + x for x in lengths]
    return Cost(aggregate_ns=aggregate, n_replicas=len(lengths), overhead_ns=overhead,
                compute_ns=aggregate + overhead, wall_ns=_makespan(chains, m.n_parallel),
                looks=looks, measured_seconds=measured_seconds)


def reduction(agent: Cost, baseline: Cost) -> dict:
    """How much cheaper `agent` is than `baseline`. Fractions, not percentages."""
    def part(a: float, b: float) -> dict:
        return {"saved": 1.0 - a / b if b > 0 else float("nan"),
                "speedup": b / a if a > 0 else float("inf")}
    return {"compute": part(agent.compute_ns, baseline.compute_ns),
            "aggregate": part(agent.aggregate_ns, baseline.aggregate_ns),
            "wall": part(agent.wall_ns, baseline.wall_ns)}


@dataclass
class RoundLog:
    """Measured cost of the rounds of a real run, kept in run.json (notes["rounds"])."""

    rounds: list = field(default_factory=list)   # {"ns": per replica, "n_replicas": int, "seconds": float}

    def add(self, ns: float, n_replicas: int, seconds: float) -> None:
        self.rounds.append({"ns": round(ns, 6), "n_replicas": n_replicas, "seconds": round(seconds, 1)})

    @property
    def seconds(self) -> float:
        return sum(r.get("seconds", 0.0) for r in self.rounds)


def of_run(state, model: CostModel | None = None) -> Cost:
    """Cost of a real run, from run.json. `state` is the dict or a RunState.

    Every replica is assumed to have the full length: run_loop lets a replica that was
    added later catch up with all earlier rounds before the next one starts.
    """
    d = state if isinstance(state, dict) else {
        "rounds_ns": state.rounds_ns, "n_replicas": state.n_replicas, "notes": state.notes,
        "md": {"equilibration_ps": state.md.equilibration_ps}}
    ns = sum(d.get("rounds_ns", []))
    n = int(d.get("n_replicas", 0) or 0)
    notes = d.get("notes") or {}
    rounds = RoundLog(notes.get("rounds", []))
    plain = of([ns] * n, model or CostModel.from_run(d), looks=len(d.get("rounds_ns", [])),
               measured_seconds=rounds.seconds)
    es = (d.get("enhanced") or {}).get("outcome") or {}
    if es.get("ns_per_replica"):
        n_es = int(((d.get("enhanced") or {}).get("settings") or {}).get("n_replicas", 0) or 0) or n
        extra = of([es["ns_per_replica"]] * n_es, model or CostModel.from_run(d))
        plain = Cost(aggregate_ns=plain.aggregate_ns + extra.aggregate_ns,
                     n_replicas=max(plain.n_replicas, extra.n_replicas),
                     overhead_ns=plain.overhead_ns,   # Enhanced sampling starts from plain MD: no new equilibration.
                     compute_ns=plain.compute_ns + extra.aggregate_ns,
                     wall_ns=plain.wall_ns + extra.wall_ns, looks=plain.looks,
                     measured_seconds=plain.measured_seconds)
    return plain
