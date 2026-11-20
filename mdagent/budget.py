"""Which way of spending simulation time reaches the asked precision for the lowest cost?

This is the question the project is about: given a system, an observable and a budget,
how should the time be split between longer trajectories and more trajectories, and when
is the result precise enough to stop? Here that question is answered by measurement
instead of by argument.

Running every strategy as real MD would cost one full simulation campaign per strategy.
Instead one expensive **pool** is simulated once: many independent replicas, each much
longer than any strategy would need. Every strategy is then **replayed** against that
pool: at each decision point it sees only the frames its own allocation would have
produced (its replicas, up to its current length), decides to run longer, to start
another replica or to stop, and pays for exactly what it used (cost.py). The pool is
plain unbiased MD, so a replay is exact for strategies that only extend trajectories or
start new independent ones. It cannot replay a strategy that restarts from a chosen frame
(adaptive sampling): those segments do not exist in the pool and have to be simulated.

The strategies:

- "fixed":        the usual plan. A fixed number of replicas, each run to the full budget,
                  with no decisions. This is the baseline the saving is measured against.
- "early_stop":   the same fixed number of replicas, but it stops at the first look where
                  every item is precise enough.
- "add_replicas": it also decides where to put the next time: one more replica when the
                  error comes mainly from the spread between replicas, otherwise longer.
- "mdagent":      the policy of this program (policy.py): early stopping, more replicas
                  only when the replicas clearly disagree and only early in the budget,
                  running past the budget when items are still moving, and stopping early
                  when an item does not change state at all, because plain MD cannot
                  answer it however long it runs.

The answer of a strategy is finally checked against an independent reference: the known
populations for a model pool, otherwise the pool replicas that the strategy did **not**
use, at their full length. Checking against the whole pool would compare a result with
itself, because the strategy's own frames are part of it. Two checks are reported
(check_against): "accurate", the error the strategy really made, which is the one that
says "same result for less computing"; and "equivalent", the stricter two one-sided test
(report.equivalent). Neither asks whether a difference is statistically significant: with
little data every difference passes that test, so a run that stopped too early would look
right exactly because it is imprecise.

Replicas are handed out in a shuffled order, so the whole comparison can be repeated with
different replicas (`repeats`). One lucky run proves nothing; what counts is how often a
strategy reaches an answer that is equivalent to the reference, and what it cost.

    python -m mdagent budget --run runs/long_1UBQ            # replay on a finished run
    python -m mdagent budget --synthetic 0.3 --repeats 20    # no GPU: a known two-state model
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import cost as cost_mod
from .diagnose import (STATE_NAMES, ResidueReport, assign_states, count_transitions, judge, occupancy,
                       populations_with_errors)
from .uncertainty import Precision, planned_looks

STRATEGIES = ("fixed", "early_stop", "add_replicas", "mdagent")


@dataclass
class Plan:
    """What every strategy is allowed to do, so that the comparison is fair."""

    replicas: int = 3               # Replicas the deciding strategies start with.
    first_ns: float = 2.0           # Length of the first round, per replica.
    extend_ns: float = 2.0          # Length of every later round, per replica.
    max_ns: float = 20.0            # Budget per replica.
    max_replicas: int = 10
    extend_factor: float = 1.0      # How far past max_ns a strategy may go (only "mdagent" uses it).
    min_flips_per_replica: int = 4
    no_flip_patience_ns: float = 10.0
    # The baseline: the plan a person would write down in advance, before seeing any data.
    # It is usually more generous than the start of an adaptive run, because it has to be safe
    # without looking. None: the same as `replicas` and `max_ns`.
    fixed_replicas: int | None = None
    fixed_ns: float | None = None

    def baseline(self) -> tuple[int, float]:
        return self.fixed_replicas or self.replicas, self.fixed_ns or self.max_ns

    def looks(self) -> int:
        return planned_looks(self.first_ns, self.extend_ns, self.max_ns, self.extend_factor)


@dataclass
class Pool:
    """Frame-by-frame states of every item, for every replica of one long run."""

    labels: list[str]
    state_names: dict[str, list[str]]
    series: dict[str, list[np.ndarray]]   # label -> one array of state indices per replica
    ns_per_frame: float
    source: str = ""
    truth: dict | None = None   # label -> {state: population}, when they are known (a model).

    @property
    def n_replicas(self) -> int:
        return len(self.series[self.labels[0]]) if self.labels else 0

    @property
    def ns_per_replica(self) -> float:
        if not self.labels:
            return 0.0
        return min(len(s) for s in self.series[self.labels[0]]) * self.ns_per_frame

    def frames(self, ns: float) -> int:
        return max(2, int(round(ns / self.ns_per_frame)))

    def report(self, label: str, replicas: list[int], ns: float, plan: Plan,
               precision: Precision | None) -> ResidueReport:
        """Judge one item from the frames the strategy has paid for, exactly as a run would."""
        n = self.frames(ns)
        states = [self.series[label][i][:n] for i in replicas]
        names = self.state_names[label]
        occ = [occupancy(s, names) for s in states]
        rep = ResidueReport(chain="", resseq=self.labels.index(label), resname="", name=label,
                            quantity="chi1" if names is STATE_NAMES else "distance",
                            transitions=[count_transitions(s) for s in states], occupancies=occ,
                            dominant=[max(o, key=o.get) for o in occ], ns_per_replica=ns)
        (rep.populations, rep.standard_error, rep.replicas_differ,
         rep.error_from_replicas) = populations_with_errors(states, names)
        judge(rep, plan.min_flips_per_replica, precision)
        return rep

    def reports(self, replicas: list[int], ns: float, plan: Plan,
                precision: Precision | None) -> list[ResidueReport]:
        return [self.report(k, replicas, ns, plan, precision) for k in self.labels]


def pool_from_run(run_dir: str | Path) -> Pool:
    """Read the states of a finished plain MD run: one series per replica, per item."""
    import mdtraj as md

    from .agent import RunState
    from .diagnose import load_run

    run_dir = Path(run_dir)
    state = RunState.load(run_dir)
    n_parts = len(state.rounds_ns)
    if not n_parts:
        raise ValueError(f"{run_dir} has no finished rounds to replay.")
    ns_per_frame = state.md.save_interval_ps / 1000.0
    if state.quantity == "distance_states":
        from .observables import distance_atoms, parse_distances, two_state

        specs = parse_distances(state.distances)
        pairs = distance_atoms(md.load_topology(str(run_dir / "protein.pdb")), specs)
        atoms = sorted({i for pair in pairs for i in pair})
        where = {a: k for k, a in enumerate(atoms)}
        trajs = load_run(run_dir, state.target, n_parts, atoms=atoms)
        labels, names, series = [], {}, {}
        for spec, (i, j) in zip(specs, pairs):
            values = [md.compute_distances(t, [(where[i], where[j])])[:, 0] for t in trajs]
            labels.append(spec.label)
            names[spec.label] = ["near", "far"]
            series[spec.label] = [two_state(v, spec.cutoff_nm, spec.buffer_nm) for v in values]
        return Pool(labels, names, series, ns_per_frame, source=str(run_dir.resolve()))

    from .diagnose import fixed_chi1_residues

    fixed = fixed_chi1_residues(run_dir / "protein.pdb")
    trajs = load_run(run_dir, state.target, n_parts)
    top = trajs[0].topology
    chi_idx, _ = md.compute_chi1(trajs[0][:1])
    wanted = set(state.target)
    keep = []
    for k, quad in enumerate(chi_idx):
        res = top.atom(int(quad[1])).residue
        key = (res.chain.chain_id, res.resSeq)
        if key in wanted and key not in fixed:
            keep.append((k, res))
    angles = [np.degrees(md.compute_chi1(t)[1]) for t in trajs]
    labels, names, series = [], {}, {}
    for k, res in keep:
        label = f"{res.name}{res.resSeq}"
        labels.append(label)
        names[label] = STATE_NAMES
        series[label] = [assign_states(a[:, k]) for a in angles]
    if not labels:
        raise ValueError(f"{run_dir} has no target side chain with a chi1 angle to replay.")
    return Pool(labels, names, series, ns_per_frame, source=str(run_dir.resolve()))


def markov_pool(populations: list[float], rate_per_ns: float = 1.0, n_replicas: int = 12,
                ns: float = 60.0, ns_per_frame: float = 0.005, seed: int = 0,
                label: str = "model") -> Pool:
    """A pool with known populations: a two- or three-state Markov chain.

    Use it to test the strategies without a GPU, and to check that the equivalence test
    finds the right answer: the true populations are given, not estimated. `rate_per_ns`
    is how often the chain leaves its current state, so 1/rate is how slow the item is.
    """
    p = np.asarray(populations, dtype=float)
    p = p / p.sum()
    names = ["s%d" % i for i in range(len(p))] if len(p) != 3 else STATE_NAMES
    rng = np.random.default_rng(seed)
    n_frames = int(round(ns / ns_per_frame))
    leave = 1.0 - math.exp(-rate_per_ns * ns_per_frame)   # Chance to try a jump in one frame.
    series = []
    for _ in range(n_replicas):
        s = np.empty(n_frames, dtype=int)
        s[0] = rng.choice(len(p), p=p)
        jumps = rng.random(n_frames) < leave
        targets = rng.choice(len(p), size=n_frames, p=p)   # Detailed balance: jump to a state with its own weight.
        for t in range(1, n_frames):
            s[t] = targets[t] if jumps[t] else s[t - 1]
        series.append(s)
    return Pool([label], {label: names}, {label: series}, ns_per_frame,
                source=f"synthetic Markov model, populations {[round(float(x), 3) for x in p]}, "
                       f"rate {rate_per_ns:g}/ns, seed {seed}",
                truth={label: {n: float(v) for n, v in zip(names, p)}})


@dataclass
class Step:
    look: int
    ns: float
    n_replicas: int
    action: str
    reason: str


@dataclass
class Trial:
    strategy: str
    stop_reason: str                      # target_reached, stalled, budget_exhausted, pool_exhausted
    ns_per_replica: float
    replicas: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)   # label -> populations, errors, verdict, met
    steps: list = field(default_factory=list)
    met: bool = False                     # Every item reached the asked precision (its own judgement).
    unresolved: list = field(default_factory=list)  # Items plain MD could not answer at all.
    check: dict = field(default_factory=dict)       # Against the held-out reference.
    accurate: bool | None = None          # The answer is really within the tolerance of the reference.
    equivalent: bool | None = None        # Stricter: the difference is proven small (TOST).


def _open_items(reports: list[ResidueReport]) -> list[ResidueReport]:
    return [r for r in reports if r.verdict not in ("ok", "no_chi1", "fixed_chi1")]


def _decide(strategy: str, reports: list[ResidueReport], ns: float, replicas: list[int],
            plan: Plan) -> tuple[str, str]:
    """One decision: "stop", "extend" or "add_replica", and why. Mirrors policy.plain_md_next."""
    from .policy import spread_labels, stalled_labels

    open_ = _open_items(reports)
    room = plan.max_ns - ns > 1e-9
    if strategy == "fixed":
        # No decisions at all: it runs the plan that was fixed in advance.
        return ("extend", "fixed plan") if plan.baseline()[1] - ns > 1e-9 else \
            ("stop", "the plan that was fixed in advance is finished")
    if not open_:
        return "stop", "every item reached the asked precision"

    stuck = stalled_labels(open_, ns, plan.no_flip_patience_ns)
    if strategy == "mdagent" and stuck and len(stuck) == len(open_):
        # Plain MD cannot answer these, however long it runs: stop instead of spending the rest.
        return "stop", f"{', '.join(stuck)} never change state, so more plain MD cannot answer them"

    moving = [r for r in open_ if sum(r.transitions) > 0]
    if strategy in ("add_replicas", "mdagent") and len(replicas) < plan.max_replicas:
        early = ns < 0.5 * plan.max_ns
        if strategy == "add_replicas":
            spread = [r.label for r in moving if r.error_from_replicas]
            if spread:
                return "add_replica", f"the error of {', '.join(spread)} comes mainly from the replica spread"
        elif early:
            spread = spread_labels(moving)
            if spread:
                return "add_replica", f"the replicas clearly disagree about {', '.join(spread)}"
    if room:
        return "extend", "items are not precise enough yet"
    if strategy == "mdagent" and moving and not stuck:
        cap = plan.extend_factor * plan.max_ns
        if ns < cap - 1e-9:
            return "extend", f"the budget is used up, but {', '.join(r.label for r in moving)} are still moving"
    return "stop", "the budget is used up"


def _stop_reason(reports: list[ResidueReport], ns: float, plan: Plan, exhausted: bool) -> str:
    open_ = _open_items(reports)
    if not open_:
        return "target_reached"
    if exhausted:
        return "pool_exhausted"
    from .policy import stalled_labels

    if stalled_labels(open_, ns, plan.no_flip_patience_ns):
        return "stalled"
    return "budget_exhausted"


def replay(pool: Pool, strategy: str, plan: Plan, precision: Precision,
           model: cost_mod.CostModel | None = None, order: list[int] | None = None) -> Trial:
    """Run one strategy against the pool and report what it cost and what it found."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}. Use one of {', '.join(STRATEGIES)}.")
    order = list(order if order is not None else range(pool.n_replicas))
    start, baseline_ns = plan.baseline() if strategy == "fixed" else (plan.replicas, plan.max_ns)
    replicas = order[:start]
    if len(replicas) < 2:
        raise ValueError("replaying needs at least 2 replicas in the pool.")
    ns = min(plan.first_ns, pool.ns_per_replica)
    cap = max(baseline_ns * (plan.extend_factor if strategy == "mdagent" else 1.0), plan.first_ns)
    trial = Trial(strategy=strategy, stop_reason="", ns_per_replica=ns)
    look, exhausted = 0, False
    while True:
        look += 1
        reports = pool.reports(replicas, ns, plan, precision)
        action, reason = _decide(strategy, reports, ns, replicas, plan)
        if action == "extend" and (ns >= pool.ns_per_replica - 1e-9 or ns >= cap - 1e-9):
            exhausted = ns >= pool.ns_per_replica - 1e-9 and ns < cap - 1e-9
            action, reason = "stop", ("the pool has no more frames" if exhausted else reason)
        if action == "add_replica" and len(replicas) >= len(order):
            action, reason = "extend", reason + " (the pool has no unused replica left)"
        trial.steps.append(Step(look, round(ns, 4), len(replicas), action, reason))
        if action == "stop":
            break
        if action == "add_replica":
            replicas = replicas + [order[len(replicas)]]
        else:
            ns = round(min(ns + plan.extend_ns, cap, pool.ns_per_replica), 6)

    open_ = _open_items(reports)
    trial.stop_reason = _stop_reason(reports, ns, plan, exhausted)
    trial.ns_per_replica = round(ns, 4)
    trial.replicas = list(replicas)
    trial.cost = cost_mod.of([ns] * len(replicas), model, looks=look).as_dict()
    trial.results = {r.label: {"populations": r.populations, "standard_error": r.standard_error,
                               "halfwidth": r.halfwidth, "transitions": r.transitions,
                               "verdict": r.verdict} for r in reports}
    trial.met = not open_
    trial.unresolved = [r.label for r in open_ if sum(r.transitions) == 0]
    return trial


def reference(pool: Pool, used: list[int], plan: Plan, precision: Precision) -> tuple[dict, list[int], bool]:
    """The answer to compare a strategy with: the known truth, or the replicas it did not use.

    For a model pool the populations are known exactly. Otherwise the reference is the
    left-over pool replicas at their full length. Comparing with the whole pool would compare
    a result with itself, because the strategy's own frames are part of it.

    Returns the populations with their errors, which replicas they come from, and whether the
    reference is independent of the strategy (it is not, if no replica was left over).
    """
    if pool.truth:
        return ({k: {"populations": dict(v), "standard_error": {s: 0.0 for s in v},
                     "transitions": [], "verdict": "known"} for k, v in pool.truth.items()},
                [], True)
    left = [i for i in range(pool.n_replicas) if i not in set(used)]
    independent = len(left) >= 2
    if not independent:
        left = list(range(pool.n_replicas))
    reports = pool.reports(left, pool.ns_per_replica, plan, precision)
    out = {r.label: {"populations": r.populations, "standard_error": r.standard_error,
                     "transitions": r.transitions, "verdict": r.verdict} for r in reports}
    return out, left, independent


def check_against(trial: Trial, ref: dict, epsilon: float, alpha: float = 0.05) -> None:
    """Compare a strategy's answer with the reference, in two ways.

    accurate:   the answer really is within the tolerance. The difference from the reference is
                the error the strategy actually made; the reference's own error is added to it,
                so the check stays honest when the reference is itself an estimate. This is the
                number that says "same result for less computing".
    equivalent: the stricter two one-sided test (report.equivalent), which also needs the
                strategy's own error to be small enough to prove that the difference is small.
                A result at the edge of the asked precision cannot pass it, so it is reported
                as extra information, not as the headline.
    """
    from .report import equivalent as tost
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - alpha)
    worst, out, accurate, ref_error = 0.0, {}, True, 0.0
    for label, got in trial.results.items():
        if label not in ref or not got.get("populations"):
            continue
        for s, p in got["populations"].items():
            pr = ref[label]["populations"].get(s, float("nan"))
            er = ref[label]["standard_error"].get(s, float("nan"))
            er = er if math.isfinite(er) else 0.0
            d = p - pr
            near = math.isfinite(d) and abs(d) + z * er <= epsilon
            _, proven = tost(p, got["standard_error"][s], pr, er, epsilon, alpha)
            out[f"{label}/{s}"] = {"difference": round(d, 4) if math.isfinite(d) else None,
                                   "accurate": near, "equivalent": proven}
            accurate = accurate and near
            ref_error = max(ref_error, er)
            if math.isfinite(d):
                worst = max(worst, abs(d))
    # The reference's own error is part of the check, so a weak reference can make a good strategy
    # look inaccurate. It is reported so that the pool can be made larger instead.
    trial.check = {"worst_difference": round(worst, 4), "reference_error_max": round(ref_error, 4), "items": out}
    trial.accurate = bool(out) and accurate
    trial.equivalent = bool(out) and all(v["equivalent"] for v in out.values())


@dataclass
class Comparison:
    plan: dict
    precision: dict
    cost_model: dict
    pool: dict
    repeats: list = field(default_factory=list)   # [{"seed": int, "trials": {strategy: Trial}}]
    summary: dict = field(default_factory=dict)   # strategy -> medians and rates


def compare_strategies(pool: Pool, plan: Plan | None = None, precision: Precision | None = None,
                       model: cost_mod.CostModel | None = None, repeats: int = 1,
                       seed: int = 0, strategies: list[str] | None = None) -> Comparison:
    """Replay every strategy on the pool, `repeats` times with different replicas."""
    plan = plan or Plan()
    precision = precision or Precision(looks=plan.looks())
    model = model or cost_mod.CostModel()
    names = list(strategies or STRATEGIES)
    out = Comparison(plan=asdict(plan), precision=asdict(precision), cost_model=asdict(model),
                     pool={"source": pool.source, "n_replicas": pool.n_replicas,
                           "ns_per_replica": round(pool.ns_per_replica, 4), "items": pool.labels,
                           "ns_per_frame": pool.ns_per_frame})
    rng = random.Random(seed)
    for k in range(max(1, repeats)):
        order = list(range(pool.n_replicas))
        if k:
            rng.shuffle(order)
        trials = {}
        for name in names:
            t = replay(pool, name, plan, precision, model, order)
            ref, used_by_ref, independent = reference(pool, t.replicas, plan, precision)
            check_against(t, ref, precision.epsilon, precision.alpha)
            t.check["reference_replicas"] = used_by_ref
            t.check["independent_reference"] = independent
            trials[name] = t
        out.repeats.append({"seed": k, "order": order,
                            "trials": {n: asdict(t) for n, t in trials.items()}})
    out.summary = _summarize(out, names)
    return out


def _median(values: list[float]) -> float:
    v = sorted(values)
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _summarize(out: Comparison, names: list[str]) -> dict:
    summary = {}
    for name in names:
        trials = [r["trials"][name] for r in out.repeats]
        compute = [t["cost"]["compute_ns"] for t in trials]
        summary[name] = {
            "repeats": len(trials),
            "compute_ns_median": round(_median(compute), 3),
            "compute_ns_range": [round(min(compute), 3), round(max(compute), 3)],
            "aggregate_ns_median": round(_median([t["cost"]["aggregate_ns"] for t in trials]), 3),
            "wall_ns_median": round(_median([t["cost"]["wall_ns"] for t in trials]), 3),
            "replicas_median": _median([t["cost"]["n_replicas"] for t in trials]),
            "ns_per_replica_median": round(_median([t["ns_per_replica"] for t in trials]), 3),
            "precision_rate": sum(bool(t["met"]) for t in trials) / len(trials),
            "accurate_rate": sum(bool(t["accurate"]) for t in trials) / len(trials),
            "equivalent_rate": sum(bool(t["equivalent"]) for t in trials) / len(trials),
            "worst_difference_max": round(max(t["check"]["worst_difference"] for t in trials), 4),
            "reference_error_max": round(max(t["check"].get("reference_error_max", 0.0) for t in trials), 4),
            "stop_reasons": {r: sum(t["stop_reason"] == r for t in trials) for r in
                             sorted({t["stop_reason"] for t in trials})},
            "unresolved": sorted({u for t in trials for u in t["unresolved"]}),
        }
    base = summary.get("fixed", {}).get("compute_ns_median")
    per_day = out.cost_model.get("ns_per_day", 0.0)
    for name, s in summary.items():
        s["saved_vs_fixed"] = (round(1 - s["compute_ns_median"] / base, 3)
                               if base else None)
        if per_day > 0:
            s["gpu_hours_median"] = round(s["compute_ns_median"] / per_day * 24.0, 2)
    return summary


def format_table(out: Comparison, zh: bool = False) -> str:
    """One line per strategy: what it cost, whether it was precise, and whether it was right."""
    s = out.summary
    head = (f"{'strategy':<14}{'compute ns':>12}{'aggregate':>11}{'wall':>8}{'replicas':>10}"
            f"{'ns each':>9}{'precise':>9}{'accurate':>10}{'proven':>8}{'worst err':>11}{'saved':>8}")
    lines = [head, "-" * len(head)]
    for name, v in s.items():
        saved = "-" if v["saved_vs_fixed"] is None else f"{v['saved_vs_fixed']:.0%}"
        lines.append(f"{name:<14}{v['compute_ns_median']:>12.1f}{v['aggregate_ns_median']:>11.1f}"
                     f"{v['wall_ns_median']:>8.1f}{v['replicas_median']:>10.1f}"
                     f"{v['ns_per_replica_median']:>9.1f}{v['precision_rate']:>8.0%}"
                     f"{v['accurate_rate']:>10.0%}{v['equivalent_rate']:>8.0%}"
                     f"{v['worst_difference_max']:>11.3f}{saved:>8}")
    repeats = s[next(iter(s))]["repeats"] if s else 0
    lines += ["",
              "compute ns = production ns of all replicas plus their equilibration (the main cost). "
              f"wall = the longest chain with {out.cost_model['n_parallel']} replica(s) running at a time.",
              f"precise  = the strategy's own judgement that it has enough data "
              f"({Precision(**out.precision).describe()}).",
              f"accurate = its answer really is within {out.precision['epsilon']:g} of the reference. "
              "This is the one that matters: cheaper only counts if it is still right.",
              "proven   = the stricter two one-sided test, which also needs the strategy's own error to be small "
              "enough to prove the agreement.",
              f"worst err = largest difference from the reference, over every state and repeat. "
              f"Shares are over {repeats} repeat(s).",
              "saved = less compute than the fixed plan. A negative number means it cost more."]
    if zh:
        lines.append("compute ns 是所有重复的生产时间加上它们的平衡时间（主要成本）；wall 是等待时间，按同时跑的"
                     "重复数折算；precise 是策略自己判断数据够了；accurate 是结果真的在容差内（这一列才说明"
                     "“更省但同样正确”）；proven 是更严格的等效性检验；saved 是比固定方案省下的计算量。")
    return "\n".join(lines)


def write(out: Comparison, out_dir: Path, name: str = "budget") -> list[Path]:
    """Write <name>.json and <name>.md next to the pool."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.json", out_dir / f"{name}.md"]
    paths[0].write_text(json.dumps(asdict(out), indent=2, default=str))
    p = Precision(**out.precision)
    L = [f"# Budget comparison: how to spend simulation time ({name})", "",
         f"Written {time.strftime('%Y-%m-%d %H:%M')}.", "",
         "## What was compared", "",
         f"- Pool: {out.pool['source']} — {out.pool['n_replicas']} replicas of "
         f"{out.pool['ns_per_replica']:g} ns, items: {', '.join(out.pool['items'])}.",
         f"- Precision asked for: {p.describe()}.",
         f"- Plan every strategy had to keep: {out.plan['replicas']} replicas at the start, rounds of "
         f"{out.plan['extend_ns']:g} ns, budget {out.plan['max_ns']:g} ns per replica, at most "
         f"{out.plan['max_replicas']} replicas.",
         f"- Cost model: {out.cost_model['equilibration_ns']:g} ns of equilibration per replica, "
         f"{out.cost_model['n_parallel']} replica(s) at a time.",
         f"- Repeats: {len(out.repeats)}, each with a different set of replicas.", "",
         "## Result", "", "```", format_table(out), "```", "",
         "## How to read it", "",
         "A strategy is only better if it is both cheaper and still right. `precise` is the strategy's own "
         "judgement that it has enough data; `accurate` is the check against "
         + ("the known populations of the model" if "Markov model" in out.pool["source"] else
            "the pool replicas it did not use")
         + ", so it is the one that matters. A strategy that is cheap but not accurate stopped too early.", "",
         "`unresolved` items never changed state within the time the strategy ran. Plain MD cannot give their "
         "populations at all, so stopping early on them saves time without losing an answer; they need enhanced "
         "sampling or a restart strategy instead.", ""]
    if out.cost_model.get("ns_per_day", 0.0) > 0:
        L += [f"At {out.cost_model['ns_per_day']:g} ns per day on this computer, the median cost in GPU-hours is: "
              + ", ".join(f"{k} {v['gpu_hours_median']:.1f}" for k, v in out.summary.items()) + ".", ""]
    worst_ref = max((v.get("reference_error_max", 0.0) for v in out.summary.values()), default=0.0)
    if worst_ref > 0:
        L += [f"The reference has an error of its own, at most {worst_ref:.3f}. It is counted against the "
              f"tolerance of {out.precision['epsilon']:g}, so a weak reference can make a good strategy look "
              "inaccurate. If it is not much smaller than the tolerance, give the pool more replicas or more "
              "length rather than believing the `accurate` column.", ""]
    for name_, v in out.summary.items():
        L.append(f"- {name_}: stop reasons {v['stop_reasons']}"
                 + (f", unresolved: {', '.join(v['unresolved'])}" if v["unresolved"] else "")
                 + f", largest difference from the reference {v['worst_difference_max']:.3f}.")
    L += ["", "## Limits", "",
          "- The replay is exact for strategies that only run longer or start another independent replica. "
          "A strategy that restarts from a chosen frame cannot be replayed: those frames do not exist in the pool.",
          "- The pool is one system with one force field at one temperature. A saving measured here does not "
          "carry over to a system with slower motions by itself.",
          "- The reference is the pool's own left-over replicas, which are finite. If the reference is not "
          "independent (too few replicas left), the comparison says so in the json.",
          "- The repeats are not fully independent experiments: they take different replicas out of the same pool. "
          "They show how much the result depends on which replicas you happen to get, which is what matters for "
          "deciding whether a saving is real, but they cannot stand in for repeating the whole campaign.", ""]
    paths[1].write_text("\n".join(L) + "\n", encoding="utf-8")
    return paths
