"""Experiment A: task-aware budget allocation for unbiased MD.

The question: given an MD task and a precision fixed in advance, can an agent decide how many
replicas to run, how long each one runs, and when to stop, so that the result is statistically
reliable for the least computing? This is a time-to-accuracy benchmark, not a convergence check.

Two criteria are kept apart, because mixing them lets an agent "learn" the answer:

- The ONLINE stopping criterion uses only the trajectories the method has produced: populations,
  their autocorrelation-aware errors (block averaging), calibrated intervals and transition counts.
  It never sees the reference.
- The OFFLINE success criterion is applied afterwards: the answer is compared with an independent
  in-house reference (pool replicas no method used, same force field, water, temperature and
  protocol). It shows whether a method stopped too early.

Arms (all plain, unbiased MD; a new replica always starts from the same prepared structure with a
new seed, never from a chosen frame, which would be adaptive sampling):

- fixed_n: n replicas of a fixed length, no decisions. Scanned over lengths, so that each family's
  own cost to reach the same accuracy and reliability can be read off (Fixed-Long is n=1,
  Fixed-Parallel is n=4 or 8).
- sequential_ci: n replicas, a look every `chunk_ns`, STOP when every population's calibrated
  interval is narrow enough and sampling is adequate. This is the conventional statistical stopping
  rule and the baseline the agent has to beat: if the agent is not cheaper than it, the agent adds
  nothing over textbook sequential stopping.
- agent: decides at each look whether to stop, to run longer or to add replicas, from the error
  decomposition (inside vs between replicas) and a projection of the cost still needed; it gives up
  early with INSUFFICIENT_EVIDENCE when the projection is far beyond the budget or nothing moves.

Observables (each judged separately; a method may stop on the primary one and must then say which
others are not converged rather than claim them):

- populations of the basins (primary), max |p - p*| <= population tolerance;
- free-energy difference of the two main basins, |dG - dG*| <= dg tolerance (kT);
- transition rates between the two main basins, relative error <= rate tolerance;
- the angle landscape (histogram), Jensen-Shannon divergence on the well-supported bins.

Per method and over many trials (different replicas each time): success rate, false-stop rate
(STOP but wrong), honest INSUFFICIENT_EVIDENCE rate, coverage of the reported intervals, cost at
stop (C_stop) and the cost at which the answer first became and stayed accurate along the same path
(C_hit), GPU-hours and wall-clock.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np

from . import cost as cost_mod
from .uncertainty import block_standard_error, error_parts

STOP, INSUFFICIENT = "STOP", "INSUFFICIENT_EVIDENCE"


@dataclass
class Targets:
    """Success thresholds, fixed before any data exist. Candidate values, not literature standards."""

    population: float = 0.05       # max |p - p*|, and the online interval half-width
    dg_kt: float = 0.5             # |dG - dG*| in kT
    rate_rel: float = 0.2          # relative error of a rate
    landscape_js: float = 0.05     # Jensen-Shannon divergence in bits, on well-supported bins
    alpha: float = 0.05
    reliability: float = 0.9       # share of trials that must succeed
    false_stop_max: float = 0.1    # largest acceptable share of STOPs with a wrong answer
    min_transitions: int = 4       # per replica, before an occupancy estimate counts as sampled


@dataclass
class Design:
    """What every method may do. The same for all arms, so the comparison is fair."""

    replicas: int = 4              # the agent starts here
    sequential: tuple = (4, 8)     # Sequential-CI arms: this many replicas each, fixed
    chunk_ns: float = 5.0          # sequential_ci looks every chunk_ns; also the agent's first look
    max_ns: float = 100.0          # per replica (the pool length)
    budget_ns: float = 800.0       # production ns, all replicas together, for the adaptive arms
    max_replicas: int = 8
    agent_looks: int = 3           # the agent's intervals are calibrated for at most this many looks
    families: tuple = (1, 4, 8)    # fixed plans: Fixed-Long (1) and Fixed-Parallel (4, 8)
    scan_ns: float = 5.0           # fixed plans are scanned in steps of this length
    burn_in_ns: float = 0.0        # left out of every estimate (still paid for)
    patience_ns: float = 20.0      # no transition at all after this: plain MD cannot answer
    give_up_factor: float = 3.0    # projected cost above this many budgets: give up now
    margin: float = 1.2            # the agent asks for 20% more than the projection
    final_margin: float = 1.6      # before its last look there is no second chance: ask for more

    def sequential_looks(self, n: int) -> int:
        return max(1, int(math.ceil(min(self.max_ns, self.budget_ns / n) / self.chunk_ns)))


def z_bonferroni(alpha: float, looks: int) -> float:
    """Two-sided interval factor with alpha shared over a known largest number of looks."""
    return NormalDist().inv_cdf(1 - alpha / (2 * max(1, looks)))


# ---------------------------------------------------------------------------------------------
# The data: frame-by-frame basin states and angles of every pool replica.

@dataclass
class BenchPool:
    labels: list
    names: dict          # label -> basin names
    states: dict         # label -> [array of state indices, one per replica]
    values: dict         # label -> [array (frames, dims) of angles in degrees, one per replica]
    ns_per_frame: float
    equilibration_ns: float = 0.6
    source: str = ""
    truth: dict | None = None   # label -> populations, for a model

    @property
    def n_replicas(self) -> int:
        return len(self.states[self.labels[0]])

    @property
    def ns_per_replica(self) -> float:
        return min(len(s) for s in self.states[self.labels[0]]) * self.ns_per_frame

    def frames(self, ns: float) -> int:
        return max(2, int(round(ns / self.ns_per_frame)))


def pool_from_run(run_dir, observables: str) -> BenchPool:
    """Read basin states and angles of a finished plain MD run (all replicas, all rounds)."""
    import mdtraj as md

    from .agent import RunState
    from .basins import angles, assign, dihedral_atoms, parse_sets
    from .diagnose import load_run

    run_dir = Path(run_dir)
    state = RunState.load(run_dir)
    sets = parse_sets(observables)
    if not sets:
        raise ValueError("give at least one observable, for example 'turn:A:52' or 'rama:A:73'.")
    top = md.load_topology(str(run_dir / "protein.pdb"))
    quads = {s.label: dihedral_atoms(top, s) for s in sets}
    atoms = sorted({a for q in quads.values() for quad in q for a in quad})
    where = {a: k for k, a in enumerate(atoms)}
    trajs = load_run(run_dir, state.target, len(state.rounds_ns), atoms=atoms)
    out = BenchPool([], {}, {}, {}, state.md.save_interval_ps / 1000.0,
                    equilibration_ns=state.md.equilibration_ps / 1000.0, source=str(run_dir.resolve()))
    for s in sets:
        local = [[where[a] for a in quad] for quad in quads[s.label]]
        vals = [angles(t, local) for t in trajs]
        out.labels.append(s.label)
        out.names[s.label] = s.names
        out.values[s.label] = vals
        out.states[s.label] = [assign(v, s) for v in vals]
    return out


def model_pool(items: list, n_replicas: int = 16, ns: float = 100.0, ns_per_frame: float = 0.01,
               seed: int = 0, spread_deg: float = 15.0) -> BenchPool:
    """A pool with known populations, for dry runs without a GPU.

    items: (populations, rate per ns) per observable. Two states use the turn centres, three the
    Ramachandran centres; angles scatter around each centre by spread_deg. The chain jumps with
    probability rate*dt per frame to a state drawn from the populations (detailed balance).
    """
    from .basins import RAMA_CENTRES, TURN_CENTRES

    rng = np.random.default_rng(seed)
    n = int(round(ns / ns_per_frame))
    out = BenchPool([], {}, {}, {}, ns_per_frame, source=f"model, seed {seed}: " + "; ".join(
        f"populations {list(p)} rate {r:g}/ns" for p, r in items), truth={})
    for k, (pops, rate) in enumerate(items):
        p = np.asarray(pops, float) / np.sum(pops)
        centres = TURN_CENTRES if len(p) == 2 else RAMA_CENTRES
        names = list(centres)[:len(p)]
        label = f"model{k + 1}"
        leave = 1 - math.exp(-rate * ns_per_frame)
        sts, vals = [], []
        for _ in range(n_replicas):
            s = np.empty(n, dtype=int)
            s[0] = rng.choice(len(p), p=p)
            jump, to = rng.random(n) < leave, rng.choice(len(p), size=n, p=p)
            for t in range(1, n):
                s[t] = to[t] if jump[t] else s[t - 1]
            c = np.array([centres[names[i]] for i in s])
            v = (c + rng.normal(0, spread_deg, c.shape) + 180) % 360 - 180
            sts.append(s)
            vals.append(v)
        out.labels.append(label)
        out.names[label] = names
        out.states[label] = sts
        out.values[label] = vals
        out.truth[label] = {nm: float(x) for nm, x in zip(names, p)}
    return out


def select_items(pool: BenchPool, min_transitions: int = 4, at_most: int = 5) -> tuple[list, dict]:
    """The rule that picks the A-Easy items from a PILOT run, never from the pool.

    An item is eligible when it changed basin at least `min_transitions` times in every pilot replica
    (the same rule the methods use to call sampling adequate). If more are eligible than `at_most`, the
    ones with the fewest transitions are kept: they are the slowest of the fast ones, so they are the
    most informative about when to stop. Returns the chosen labels and every item's transitions.
    """
    counts = {}
    for k in pool.labels:
        counts[k] = []
        for s in pool.states[k]:
            s = s[s >= 0]
            counts[k].append(int(np.count_nonzero(s[1:] != s[:-1])))
    eligible = [k for k, c in counts.items() if c and min(c) >= min_transitions]
    chosen = sorted(eligible, key=lambda k: (min(counts[k]), k))[:at_most]
    return [k for k in pool.labels if k in chosen], counts


# ---------------------------------------------------------------------------------------------
# Estimates from the frames a method has paid for.

def _hist(vals: list, bins: int = 36) -> np.ndarray:
    v = np.concatenate([np.asarray(x).reshape(len(x), -1) for x in vals]) if vals else np.zeros((0, 1))
    if v.shape[1] == 1:
        h, _ = np.histogram(v[:, 0], bins=bins, range=(-180, 180))
    else:
        h, _, _ = np.histogram2d(v[:, 0], v[:, 1], bins=bins, range=[[-180, 180], [-180, 180]])
    h = h.astype(float).ravel()
    return h / h.sum() if h.sum() else h


def js_bits(p: np.ndarray, q: np.ndarray, support: np.ndarray | None = None) -> float:
    """Jensen-Shannon divergence in bits (0 to 1), optionally only on the given bins."""
    if support is not None:
        p, q = p[support], q[support]
    if p.sum() <= 0 or q.sum() <= 0:
        return 1.0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))
    return 0.5 * kl(p) + 0.5 * kl(q)


def estimate(pool: BenchPool, label: str, replicas: list, ns: float, design: Design) -> dict:
    """Every observable of one item, from `replicas` up to `ns`, with errors that allow for time
    correlation. Nothing here looks at a reference."""
    n, b = pool.frames(ns), pool.frames(design.burn_in_ns) if design.burn_in_ns > 0 else 0
    names = pool.names[label]
    sts = [pool.states[label][i][:n] for i in replicas]
    use = [s[b:] for s in sts]
    pops, se, within, between = {}, {}, {}, {}
    for k, name in enumerate(names):
        hits = [(s[s >= 0] == k).astype(float) for s in use]
        vals = [float(h.mean()) if h.size else 0.0 for h in hits]
        errs = [block_standard_error(h) for h in hits]
        w, bt = error_parts(vals, errs)
        pops[name] = float(np.mean(vals))
        se[name] = max(w, bt) if len(vals) > 1 else (errs[0] if np.isfinite(errs[0]) else float("nan"))
        within[name], between[name] = w, bt
    # Transitions between consecutive distinct states, and the time spent in each state.
    trans = []
    counts = np.zeros((len(names), len(names)))
    dwell = np.zeros(len(names))
    for s in use:
        s = s[s >= 0]
        change = np.nonzero(s[1:] != s[:-1])[0]
        trans.append(int(len(change)))
        for i in change:
            counts[s[i], s[i + 1]] += 1
        for k in range(len(names)):
            dwell[k] += np.sum(s == k) * pool.ns_per_frame
    a, c = 0, 1   # the two main basins: the first two of the definition
    pa, pc = pops[names[a]], pops[names[c]]
    dg = -math.log(pc / pa) if pa > 0 and pc > 0 else float("inf")
    dg_se = (se[names[a]] / pa + se[names[c]] / pc) if pa > 0 and pc > 0 else float("inf")
    rates = {}
    for i, j in ((a, c), (c, a)):
        nij = counts[i, j]
        rates[f"{names[i]}->{names[j]}"] = {"k_per_ns": float(nij / dwell[i]) if dwell[i] > 0 else 0.0,
                                            "n": int(nij), "rel_se": 1 / math.sqrt(nij) if nij else float("inf")}
    vals = [pool.values[label][i][b:n] for i in replicas]
    hist = _hist(vals)
    halves = ([vals[i] for i in range(0, len(vals), 2)], [vals[i] for i in range(1, len(vals), 2)]) \
        if len(vals) > 1 else ([vals[0][: len(vals[0]) // 2]], [vals[0][len(vals[0]) // 2:]])
    split_js = js_bits(_hist(halves[0]), _hist(halves[1]))
    worst = max(names, key=lambda k: se[k] if np.isfinite(se[k]) else 1.0)
    return {"populations": pops, "se": se, "transitions": trans, "dg_kt": dg, "dg_se": dg_se,
            "rates": rates, "hist": hist, "split_js": split_js,
            "within": within[worst], "between": between[worst]}


def judge_online(est: dict, z: float, targets: Targets) -> dict:
    """The method's own verdict on each observable, from its own data only."""
    pops_ok = all(np.isfinite(v) and z * v <= targets.population for v in est["se"].values())
    adequate = bool(est["transitions"]) and min(est["transitions"]) >= targets.min_transitions
    rates_ok = all(r["n"] > 0 and z * r["rel_se"] <= targets.rate_rel for r in est["rates"].values())
    return {"populations": pops_ok and adequate, "adequate": adequate,
            "dg": adequate and np.isfinite(est["dg_se"]) and z * est["dg_se"] <= targets.dg_kt,
            "rates": adequate and rates_ok,
            "landscape": adequate and est["split_js"] <= targets.landscape_js}


def compare_to_reference(est: dict, ref: dict, z: float, targets: Targets) -> dict:
    """Offline: the real error of each observable, and whether the reported interval covered it."""
    d = {k: abs(est["populations"][k] - ref["populations"][k]) for k in ref["populations"]}
    covered = {k: (np.isfinite(est["se"][k]) and d[k] <= z * est["se"][k]) for k in d}
    dg_err = abs(est["dg_kt"] - ref["dg_kt"]) if np.isfinite(est["dg_kt"]) and np.isfinite(ref["dg_kt"]) else float("inf")
    rate_err = max((abs(r["k_per_ns"] - ref["rates"][k]["k_per_ns"]) / ref["rates"][k]["k_per_ns"]
                    if ref["rates"][k]["k_per_ns"] > 0 else float("inf")) for k, r in est["rates"].items())
    support = ref["hist"] >= 1e-3
    js = js_bits(est["hist"], ref["hist"], support)
    return {"population_error": max(d.values()), "populations": max(d.values()) <= targets.population,
            "covered": all(covered.values()), "covered_share": sum(covered.values()) / len(covered),
            "dg_error_kt": dg_err, "dg": dg_err <= targets.dg_kt,
            "rate_error": rate_err, "rates": rate_err <= targets.rate_rel,
            "landscape_js": js, "landscape": js <= targets.landscape_js}


def reference(pool: BenchPool, held: list, design: Design, targets: Targets) -> dict:
    """The in-house reference: held-out replicas at full length. Also says whether it is good enough
    to judge others: its own interval must be at most half of each tolerance."""
    z = NormalDist().inv_cdf(1 - targets.alpha / 2)
    out = {}
    for label in pool.labels:
        r = estimate(pool, label, held, pool.ns_per_replica, design)
        moved = min(r["transitions"]) >= targets.min_transitions   # trapped replicas look precise, but are not
        adequate = {
            "populations": moved and all(np.isfinite(v) and z * v <= targets.population / 2 for v in r["se"].values()),
            "dg": moved and np.isfinite(r["dg_se"]) and z * r["dg_se"] <= targets.dg_kt / 2,
            "rates": moved and all(x["n"] > 0 and z * x["rel_se"] <= targets.rate_rel / 2 for x in r["rates"].values()),
            "landscape": moved and r["split_js"] <= targets.landscape_js / 2}
        out[label] = {**r, "reference_adequate": adequate}
    return out


# ---------------------------------------------------------------------------------------------
# The arms.

OBSERVABLES = ("populations", "dg", "rates", "landscape")


@dataclass
class Trial:
    arm: str
    decision: str = ""
    reason: str = ""
    replicas: int = 0
    ns_per_replica: float = 0.0
    looks: int = 0
    cost: dict = field(default_factory=dict)
    path: list = field(default_factory=list)       # [(compute_ns, population_error, accurate)] per look
    online: dict = field(default_factory=dict)     # observable -> converged (all items)
    offline: dict = field(default_factory=dict)    # observable -> accurate (all items)
    population_error: float = float("nan")
    covered: bool = False
    covered_share: float = float("nan")
    outcome: str = ""                               # correct_stop, false_stop, correct_insufficient, over_cautious
    c_hit: float | None = None
    steps: list = field(default_factory=list)


def _snapshot(pool, order, n, ns, design, targets, ref, z):
    ests = {k: estimate(pool, k, order[:n], ns, design) for k in pool.labels}
    online = {k: judge_online(e, z, targets) for k, e in ests.items()}
    offline = {k: compare_to_reference(e, ref[k], z, targets) for k, e in ests.items()}
    return ests, online, offline


def _close(trial, pool, design, targets, ests, online, offline, n, ns, looks, model):
    trial.replicas, trial.ns_per_replica, trial.looks = n, round(ns, 4), looks
    trial.cost = cost_mod.of([ns] * n, model, looks=looks).as_dict()
    trial.online = {o: all(online[k][o] for k in pool.labels) for o in OBSERVABLES}
    trial.offline = {o: all(offline[k][o] for k in pool.labels) for o in OBSERVABLES}
    trial.population_error = max(offline[k]["population_error"] for k in pool.labels)
    trial.covered = all(offline[k]["covered"] for k in pool.labels)
    trial.covered_share = float(np.mean([offline[k]["covered_share"] for k in pool.labels]))
    right = trial.offline["populations"]
    trial.outcome = (("correct_stop" if right else "false_stop") if trial.decision == STOP
                     else ("over_cautious" if right else "correct_insufficient"))
    # C_hit along this method's own path: the first look from which the answer was accurate and stayed so.
    hit = None
    for i in range(len(trial.path)):
        if all(p[2] for p in trial.path[i:]):
            hit = trial.path[i][0]
            break
    trial.c_hit = hit


def run_sequential_ci(pool, order, design, targets, ref, model, n: int) -> Trial:
    z = z_bonferroni(targets.alpha, design.sequential_looks(n))
    ns, looks, t = design.chunk_ns, 0, Trial(f"sequential_ci_{n}")
    cap = min(design.max_ns, design.budget_ns / n, pool.ns_per_replica)
    while True:
        looks += 1
        ests, online, offline = _snapshot(pool, order, n, ns, design, targets, ref, z)
        t.path.append((cost_mod.of([ns] * n, model, looks=looks).compute_ns,
                       max(offline[k]["population_error"] for k in pool.labels),
                       all(offline[k]["populations"] for k in pool.labels)))
        if all(online[k]["populations"] for k in pool.labels):
            t.decision, t.reason = STOP, "every population interval is narrow enough and sampling is adequate"
            break
        if ns >= cap - 1e-9:
            t.decision, t.reason = INSUFFICIENT, "the budget is used up before the populations were precise"
            break
        ns = round(min(ns + design.chunk_ns, cap), 6)
    _close(t, pool, design, targets, ests, online, offline, n, ns, looks, model)
    return t


def chi2_quantile(p: float, k: int) -> float:
    """Wilson-Hilferty approximation of the chi-square quantile (good to a few percent for k >= 2)."""
    zp = NormalDist().inv_cdf(p)
    return k * (1 - 2 / (9 * k) + zp * math.sqrt(2 / (9 * k))) ** 3


def allocate(ests: dict, labels: list, n: int, ns: float, z: float, targets: Targets, design: Design,
             max_n: int, max_ns: float, equil_ns: float):
    """The cheapest (replicas, ns per replica) that is projected to reach the precision, or None.

    The error of a population is split into the part inside the replicas, which falls with all the
    data together (replicas x length), and the extra spread between replicas beyond that, which only
    more replicas reduce (for example memory of the starting structure, or a slow process that some
    replicas cross and others do not). Every extra replica also pays its equilibration, so a longer
    run is preferred unless the spread between replicas makes it useless. Lengths cannot shrink.
    Returns (replicas, ns per replica, projected compute ns).
    """
    best = None
    # Spread between replicas only counts when it is larger than chance: with n replicas and no real
    # difference between them, (between/within)^2 follows chi2(n-1)/(n-1); its 95% point is the limit.
    limit = chi2_quantile(0.95, n - 1) / (n - 1) if n > 1 else float("inf")
    for m in range(n, max_n + 1):
        need = 0.0
        for k in labels:
            w, b = ests[k]["within"], ests[k]["between"]
            real = w > 0 and (b * b) / (w * w) > limit
            extra = (b * b - w * w) * n / m if real else 0.0   # replica-to-replica part, shrinks with replicas only
            room = (targets.population / z) ** 2 - extra       # what is left for the inside part
            if room <= 0:
                need = float("inf")
                break
            need = max(need, ns * (w * w * n / m) / room)     # length at which the inside part fits
        length = max(ns, need)
        if length > max_ns + 1e-9:
            continue
        cost = m * (length + equil_ns)
        if best is None or cost < best[2]:
            best = (m, length, cost)
    return best


def run_agent(pool, order, design, targets, ref, model) -> Trial:
    """The agent's allocation rule. Frozen before the pool is analysed; see the module docstring.

    The first look only plans: it may not stop, so it spends none of alpha. After it, the agent may stop
    at no more than `agent_looks` looks, and its intervals are calibrated for exactly that many. Between
    looks it jumps to the cheapest allocation projected to reach the precision (allocate), so it needs
    few looks; before the last one it asks for a larger margin, because there is no second chance.
    """
    z = z_bonferroni(targets.alpha, design.agent_looks)
    n, ns, looks, tests, t = design.replicas, design.chunk_ns, 0, 0, Trial("agent")
    max_n = min(design.max_replicas, len(order))
    max_ns = min(design.max_ns, pool.ns_per_replica)
    while True:
        looks += 1
        planning = looks == 1
        ests, online, offline = _snapshot(pool, order, n, ns, design, targets, ref, z)
        t.path.append((cost_mod.of([ns] * n, model, looks=looks).compute_ns,
                       max(offline[k]["population_error"] for k in pool.labels),
                       all(offline[k]["populations"] for k in pool.labels)))
        if not planning:
            tests += 1
            if all(online[k]["populations"] for k in pool.labels):
                t.decision, t.reason = STOP, "every population interval is narrow enough and sampling is adequate"
                break
        spent = n * ns
        full = spent >= design.budget_ns - 1e-9 or (ns >= max_ns - 1e-9 and n >= max_n)
        if tests >= design.agent_looks or full:
            t.decision = INSUFFICIENT
            t.reason = ("the largest number of looks is reached" if tests >= design.agent_looks
                        else "the budget is used up")
            break
        stuck = [k for k in pool.labels if sum(ests[k]["transitions"]) == 0]
        if stuck and ns >= design.patience_ns:
            t.decision = INSUFFICIENT
            t.reason = f"{', '.join(stuck)} never changed basin in {ns:g} ns per replica; plain MD cannot answer"
            break
        last = tests == design.agent_looks - 1          # the next look is the last one that may stop
        margin = design.final_margin if last else design.margin
        adequate = all(online[k]["adequate"] for k in pool.labels)
        new_n, new_ns = n, ns
        if not adequate:
            # Too few transitions to trust any error estimate: run longer, doubling (more before the last look).
            new_ns = min(max_ns, (4 if last else 2) * ns)
            why = "too few transitions to trust the errors; running longer"
        else:
            plan = allocate(ests, pool.labels, n, ns, z, targets, design, max_n, max_ns, pool.equilibration_ns)
            # A projection from very little data is too noisy to give up on: wait for a quarter of the budget.
            hopeless = plan is None or plan[2] > design.give_up_factor * design.budget_ns
            if hopeless and spent < design.budget_ns / 4:
                plan = (n, min(max_ns, max(2 * ns, design.budget_ns / 4 / n)), 0.0)
                hopeless = False
            if hopeless:
                t.decision = INSUFFICIENT
                need = "no allocation within the limits reaches it" if plan is None else \
                    f"the projected cost is {plan[2]:.0f} ns"
                t.reason = (f"{need}; that is more than {design.give_up_factor:g} times the budget "
                            f"({design.budget_ns:g} ns), so more plain MD will not reach the precision")
                break
            new_n, new_ns = plan[0], min(max_ns, plan[1] * margin)
            why = (f"projected: {new_n} replicas of {plan[1]:.1f} ns reach the precision for the least compute "
                   f"(plus a {margin - 1:.0%} margin)")
        if new_n * new_ns > design.budget_ns:   # stay inside the budget
            new_ns = max(ns, design.budget_ns / new_n)
            if new_n * new_ns > design.budget_ns + 1e-9:
                new_n = n
                new_ns = min(max_ns, design.budget_ns / n)
        if new_n == n and new_ns <= ns + 1e-9:
            new_ns = min(max_ns, ns + design.chunk_ns)
        t.steps.append({"look": looks, "planning": planning, "replicas": n, "ns": round(ns, 3),
                        "next_replicas": new_n, "next_ns": round(new_ns, 3), "why": why})
        n, ns = new_n, round(new_ns, 6)
    _close(t, pool, design, targets, ests, online, offline, n, ns, looks, model)
    return t


def scan_fixed(pool, order, n, design, targets, ref, model) -> list:
    """A fixed plan of n replicas at every scanned length: (length, compute ns, population error, accurate)."""
    z = NormalDist().inv_cdf(1 - targets.alpha / 2)
    rows, ns = [], design.scan_ns
    top = min(design.max_ns, pool.ns_per_replica)
    while ns <= top + 1e-9:
        ests = {k: estimate(pool, k, order[:n], ns, design) for k in pool.labels}
        off = {k: compare_to_reference(e, ref[k], z, targets) for k, e in ests.items()}
        rows.append((round(ns, 4), cost_mod.of([ns] * n, model).compute_ns,
                     max(o["population_error"] for o in off.values()), all(o["populations"] for o in off.values())))
        ns += design.scan_ns
    return rows


# ---------------------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n <= 0:
        return [float("nan"), float("nan")]
    p = k / n
    d, c = 1 + z * z / n, p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (c - h) / d), 3), round(min(1.0, (c + h) / d), 3)]


def _q(values, q):
    v = sorted(x for x in values if x is not None and np.isfinite(x))
    return float(np.quantile(v, q)) if v else float("nan")


def run(pool: BenchPool, design: Design | None = None, targets: Targets | None = None, trials: int = 50,
        holdout: int = 6, seed: int = 0, n_parallel: int = 1, ns_per_day: float = 0.0) -> dict:
    """The whole experiment: reference, every arm over `trials` replica draws, summaries and the verdict."""
    design, targets = design or Design(), targets or Targets()
    model = cost_mod.CostModel(equilibration_ns=pool.equilibration_ns, n_parallel=n_parallel, ns_per_day=ns_per_day)
    if holdout < 2:
        raise ValueError("the reference needs at least 2 held-out replicas.")
    held = sorted(random.Random(seed).sample(range(pool.n_replicas), holdout))
    usable = [i for i in range(pool.n_replicas) if i not in held]
    need = max(design.replicas, *design.families, *design.sequential)
    if len(usable) < need:
        raise ValueError(f"{len(usable)} pool replicas are left for the methods, but a method needs {need}. "
                         "Use a larger pool or a smaller --holdout.")
    design.max_replicas = min(design.max_replicas, len(usable))
    ref = reference(pool, held, design, targets)
    rng = random.Random(seed + 1)
    adaptive = {**{f"sequential_ci_{n}": [] for n in design.sequential}, "agent": []}
    fixed = {n: [] for n in design.families}
    for k in range(trials):
        order = list(usable)
        rng.shuffle(order)
        for n in design.sequential:
            adaptive[f"sequential_ci_{n}"].append(run_sequential_ci(pool, order, design, targets, ref, model, n))
        adaptive["agent"].append(run_agent(pool, order, design, targets, ref, model))
        for n in design.families:
            fixed[n].append(scan_fixed(pool, order, n, design, targets, ref, model))
    out = {"design": asdict(design), "targets": asdict(targets), "trials": trials, "seed": seed,
           "pool": {"source": pool.source, "replicas": pool.n_replicas, "ns_per_replica": pool.ns_per_replica,
                    "items": pool.labels, "held_out": held, "equilibration_ns": pool.equilibration_ns},
           "cost_model": asdict(model),
           "reference": {k: {"populations": v["populations"], "se": v["se"], "dg_kt": v["dg_kt"],
                             "rates": v["rates"], "transitions": v["transitions"],
                             "adequate": v["reference_adequate"],
                             "truth": (pool.truth or {}).get(k)} for k, v in ref.items()},
           "arms": {a: [asdict(t) for t in ts] for a, ts in adaptive.items()},
           "fixed": {str(n): rows for n, rows in fixed.items()}}
    out["summary"] = summarize(out)
    out["verdict"] = verdict(out)
    return out


def summarize(out: dict) -> dict:
    tg = out["targets"]
    s = {}
    for arm, ts in out["arms"].items():
        n = len(ts)
        cnt = {o: sum(t["outcome"] == o for t in ts) for o in
               ("correct_stop", "false_stop", "correct_insufficient", "over_cautious")}
        stops = cnt["correct_stop"] + cnt["false_stop"]
        compute = [t["cost"]["compute_ns"] for t in ts]
        ratio = [t["cost"]["compute_ns"] / t["c_hit"] for t in ts if t["outcome"] == "correct_stop" and t["c_hit"]]
        s[arm] = {
            "trials": n, **{o: c for o, c in cnt.items()},
            "success_rate": cnt["correct_stop"] / n, "success_interval": wilson(cnt["correct_stop"], n),
            "false_stop_rate": cnt["false_stop"] / n, "false_stop_interval": wilson(cnt["false_stop"], n),
            "false_stop_share_of_stops": cnt["false_stop"] / stops if stops else 0.0,
            "coverage": float(np.mean([t["covered"] for t in ts])),
            "compute_ns_median": _q(compute, 0.5), "compute_ns_iqr": [_q(compute, 0.25), _q(compute, 0.75)],
            "aggregate_ns_median": _q([t["cost"]["aggregate_ns"] for t in ts], 0.5),
            "wall_ns_median": _q([t["cost"]["wall_ns"] for t in ts], 0.5),
            "replicas_median": _q([t["replicas"] for t in ts], 0.5),
            "ns_per_replica_median": _q([t["ns_per_replica"] for t in ts], 0.5),
            "looks_median": _q([t["looks"] for t in ts], 0.5),
            "c_hit_median": _q([t["c_hit"] for t in ts], 0.5),
            "stop_over_hit_median": _q(ratio, 0.5),
            "claims": {o: {"claimed": sum(t["online"][o] for t in ts),
                           "claimed_but_wrong": sum(t["online"][o] and not t["offline"][o] for t in ts)}
                       for o in OBSERVABLES},
        }
        if out["cost_model"]["ns_per_day"] > 0:
            s[arm]["gpu_hours_median"] = s[arm]["compute_ns_median"] / out["cost_model"]["ns_per_day"] * 24
    fam = {}
    for n, rows_per_trial in out["fixed"].items():
        lengths = [r[0] for r in rows_per_trial[0]]
        rate = [float(np.mean([rows[i][3] for rows in rows_per_trial])) for i in range(len(lengths))]
        stable = next((i for i in range(len(lengths)) if all(r >= tg["reliability"] for r in rate[i:])), None)
        hits = []
        for rows in rows_per_trial:
            first = next((i for i in range(len(rows)) if all(r[3] for r in rows[i:])), None)
            hits.append(rows[first][1] if first is not None else None)
        fam[n] = {"lengths_ns": lengths, "accurate_rate": rate,
                  "reliable_ns": lengths[stable] if stable is not None else None,
                  "reliable_compute_ns": rows_per_trial[0][stable][1] if stable is not None else None,
                  "c_hit_median": _q(hits, 0.5),
                  "error_median": [_q([rows[i][2] for rows in rows_per_trial], 0.5) for i in range(len(lengths))],
                  "error_p10": [_q([rows[i][2] for rows in rows_per_trial], 0.1) for i in range(len(lengths))],
                  "error_p90": [_q([rows[i][2] for rows in rows_per_trial], 0.9) for i in range(len(lengths))]}
    s["fixed"] = fam
    return s


def verdict(out: dict, bootstrap: int = 4000) -> dict:
    """Hypothesis A, as written down before the data exist.

    Supported only if all three hold:
    1. Reliable: the agent's success rate (STOP and accurate) is at least the target reliability, and
       its false-stop rate is at most the allowed one.
    2. Better than every Sequential-CI arm: if that arm is reliable, the paired saving over the same
       replica draws has a 95% bootstrap interval above 0; if it is not reliable, the agent must be.
       Without this the agent adds nothing over textbook sequential stopping.
    3. Cheaper than each fixed family at the same reliability: the agent's median cost is below the
       cost a fixed plan of that family needs to be accurate in the same share of trials. A family that
       never reaches it within the pool length cannot be matched at all, which counts for the agent.
    If the reference itself is not precise enough, no accuracy claim can be made, whatever the numbers.
    """
    tg, s = out["targets"], out["summary"]
    ref_ok = all(v["adequate"]["populations"] for v in out["reference"].values())
    a = s["agent"]
    c1 = a["success_rate"] >= tg["reliability"] and a["false_stop_rate"] <= tg["false_stop_max"]
    seq = {}
    for arm in (k for k in out["arms"] if k.startswith("sequential_ci")):
        q = s[arm]
        reliable = q["success_rate"] >= tg["reliability"] and q["false_stop_rate"] <= tg["false_stop_max"]
        ag = [t["cost"]["compute_ns"] for t in out["arms"]["agent"]]
        sq = [t["cost"]["compute_ns"] for t in out["arms"][arm]]
        save = [1 - x / y for x, y in zip(ag, sq) if y > 0]
        rng = random.Random(out["seed"])
        means = sorted(sum(rng.choice(save) for _ in save) / len(save) for _ in range(bootstrap))
        interval = [round(means[int(0.025 * len(means))], 3), round(means[int(0.975 * len(means)) - 1], 3)]
        seq[arm] = {"success_rate": q["success_rate"], "false_stop_rate": q["false_stop_rate"],
                    "reliable": reliable, "saving_mean": round(float(np.mean(save)), 3),
                    "saving_interval": interval,
                    # A baseline that is not reliable cannot be matched at the same reliability: the agent is
                    # better by being right, if it is reliable itself.
                    "agent_better": (interval[0] > 0) if reliable else c1}
    c2 = bool(seq) and all(v["agent_better"] for v in seq.values())
    fam = {}
    for n, f in s["fixed"].items():
        if f["reliable_compute_ns"] is None:
            fam[n] = {"reliable_compute_ns": None, "cheaper": True,
                      "note": "this fixed plan never reaches the reliability within the pool length"}
        else:
            fam[n] = {"reliable_compute_ns": f["reliable_compute_ns"],
                      "saving": round(1 - a["compute_ns_median"] / f["reliable_compute_ns"], 3),
                      "cheaper": a["compute_ns_median"] < f["reliable_compute_ns"]}
    c3 = all(v["cheaper"] for v in fam.values())
    # The protocols people write down before running: n replicas of the full pool length (for 1UBQ,
    # one 100 ns trajectory is what Huang & MacKerell 2013 used). Information, not a condition.
    literature = {}
    for n, f in s["fixed"].items():
        if not f["lengths_ns"]:
            continue
        i = len(f["lengths_ns"]) - 1
        cost_full = out["fixed"][n][0][i][1]
        literature[f"{n}x{f['lengths_ns'][i]:g}"] = {
            "compute_ns": cost_full, "accurate_rate": f["accurate_rate"][i],
            "agent_saving": round(1 - a["compute_ns_median"] / cost_full, 3) if cost_full else None}
    failed = [why for ok, why in ((c1, "not reliable enough"),
                                  (c2, "not better than every Sequential-CI arm at the same reliability"),
                                  (c3, "not cheaper than every fixed plan at the same reliability")) if not ok]
    result = ("reference_inadequate" if not ref_ok else "supported" if not failed else "not_supported")
    return {"result": result, "failed": failed if ref_ok else ["the reference is not precise enough"],
            "reliable": c1, "vs_sequential_ci": seq, "better_than_sequential_ci": c2,
            "vs_fixed": fam, "cheaper_than_fixed": c3, "vs_literature_protocols": literature}


# ---------------------------------------------------------------------------------------------
# Output: experiment_A.json, experiment_A.md, cost_accuracy.csv and cost_accuracy.svg.

ARM_TITLES = {"agent": "MD Agent", "sequential_ci_4": "Sequential-CI, 4 replicas",
              "sequential_ci_8": "Sequential-CI, 8 replicas"}


def _arm_title(arm: str) -> str:
    if arm in ARM_TITLES:
        return ARM_TITLES[arm]
    if arm.startswith("sequential_ci_"):
        return f"Sequential-CI, {arm.split('_')[-1]} replicas"
    return arm


def curve_rows(out: dict) -> list[dict]:
    """The data behind the cost-accuracy figure, one row per point (also the table view)."""
    rows = []
    for n, f in out["summary"]["fixed"].items():
        for i, length in enumerate(f["lengths_ns"]):
            rows.append({"method": f"fixed_{n}", "replicas": int(n), "ns_per_replica": length,
                         "compute_ns": out["fixed"][n][0][i][1], "error_median": f["error_median"][i],
                         "error_p10": f["error_p10"][i], "error_p90": f["error_p90"][i],
                         "accurate_rate": f["accurate_rate"][i]})
    for arm, ts in out["arms"].items():
        for k, t in enumerate(ts):
            rows.append({"method": arm, "trial": k, "decision": t["decision"], "outcome": t["outcome"],
                         "replicas": t["replicas"], "ns_per_replica": t["ns_per_replica"],
                         "compute_ns": t["cost"]["compute_ns"], "error_median": t["population_error"]})
    return rows


def _nice_step(top: float, ticks: int = 5) -> float:
    """A round tick step (1, 2 or 5 times a power of ten) giving about `ticks` intervals up to `top`."""
    raw = max(top, 1e-12) / ticks
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)


def svg_chart(out: dict, families_shown: tuple = (1, 4, 8)) -> str:
    """Cost-accuracy figure: error against compute for fixed plans (lines) and the adaptive arms (points)."""
    W, H, L, R, T, B = 840, 440, 64, 230, 40, 52
    s = out["summary"]
    fams = [str(n) for n in families_shown if str(n) in s["fixed"]]
    arms = list(out["arms"])
    xs = [r["compute_ns"] for r in curve_rows(out) if r["method"] in [f"fixed_{n}" for n in fams] or "trial" in r]
    ys = [r["error_median"] for r in curve_rows(out) if np.isfinite(r["error_median"])]
    tol = out["targets"]["population"]
    xstep = _nice_step(max(xs) if xs else 1.0)
    xmax = xstep * math.ceil((max(xs) if xs else 1.0) * 1.02 / xstep)
    ystep = _nice_step(max(max(ys) if ys else 0.1, tol * 2))
    ymax = ystep * math.ceil(max(max(ys) if ys else 0.1, tol * 2) * 1.02 / ystep)

    def X(v):
        return L + (W - L - R) * v / xmax

    def Y(v):
        return T + (H - T - B) * (1 - min(v, ymax) / ymax)

    # Slots in fixed order: fixed plans first, then the adaptive arms (colour follows the method).
    slots = [f"fixed_{n}" for n in fams] + arms
    colour_var = {m: f"--series-{i + 1}" for i, m in enumerate(slots)}
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         'role="img" aria-labelledby="t d" font-family="system-ui, sans-serif" font-size="12">',
         '<title id="t">Cost and accuracy</title>',
         '<desc id="d">Largest population error against compute cost. Lines: fixed plans (median over trials). '
         'Points: the stop of each adaptive method in each trial; large markers are the medians.</desc>',
         '<style>svg{--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e4e3de;'
         '--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;'
         '--series-6:#008300}'
         '@media (prefers-color-scheme: dark){svg{--surface-1:#1a1a19;--text-primary:#f0efec;'
         '--text-secondary:#c3c2b7;--grid:#33332f;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;'
         '--series-4:#c98500;--series-5:#d55181;--series-6:#0ca30c}}'
         'text{fill:var(--text-secondary)} .h{fill:var(--text-primary);font-weight:600}</style>',
         f'<rect width="{W}" height="{H}" fill="var(--surface-1)"/>',
         f'<text class="h" x="{L}" y="22">Largest population error against compute (median of '
         f'{out["trials"]} trials)</text>']
    for i in range(int(round(ymax / ystep)) + 1):   # recessive grid and axis labels
        yv = ystep * i
        p.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="var(--grid)"/>')
        p.append(f'<text x="{L - 8}" y="{Y(yv) + 4:.1f}" text-anchor="end">{yv:.2f}</text>')
    for i in range(int(round(xmax / xstep)) + 1):
        xv = xstep * i
        p.append(f'<text x="{X(xv):.1f}" y="{H - B + 18}" text-anchor="middle">{xv:g}</text>')
    p.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 12}" text-anchor="middle">compute (ns of MD, '
             'equilibration included)</text>')
    p.append(f'<text x="16" y="{(T + H - B) / 2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 16 {(T + H - B) / 2:.0f})">max |p − p*|</text>')
    p.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(tol):.1f}" y2="{Y(tol):.1f}" stroke="var(--text-secondary)" '
             'stroke-dasharray="4 4"/>')
    p.append(f'<text x="{W - R + 6}" y="{Y(tol) + 4:.1f}">tolerance {tol:g}</text>')
    for n in fams:
        f, c = s["fixed"][n], f"var({colour_var[f'fixed_{n}'] })"
        pts = [(out["fixed"][n][0][i][1], e) for i, e in enumerate(f["error_median"]) if np.isfinite(e)]
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        p.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>')
        for (x, y), length, rate in zip(pts, f["lengths_ns"], f["accurate_rate"]):
            p.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="6" fill="transparent">'
                     f'<title>{n} replica(s) × {length:g} ns: {x:.0f} ns compute, median error {y:.3f}, '
                     f'accurate in {rate:.0%} of trials</title></circle>')
        x, y = pts[-1]
        p.append(f'<text x="{X(x) + 8:.1f}" y="{Y(y) + 4:.1f}">fixed {n}×</text>')
    for arm in arms:
        c = f"var({colour_var[arm]})"
        ts = out["arms"][arm]
        for t in ts:
            if np.isfinite(t["population_error"]):
                p.append(f'<circle cx="{X(t["cost"]["compute_ns"]):.1f}" cy="{Y(t["population_error"]):.1f}" r="3" '
                         f'fill="{c}" fill-opacity="0.35"><title>{_arm_title(arm)}: {t["decision"]} at '
                         f'{t["replicas"]} × {t["ns_per_replica"]:g} ns, error {t["population_error"]:.3f}'
                         f'</title></circle>')
        cx, cy = s[arm]["compute_ns_median"], _q([t["population_error"] for t in ts], 0.5)
        if np.isfinite(cx) and np.isfinite(cy):
            shape = (f'<rect x="{X(cx) - 5:.1f}" y="{Y(cy) - 5:.1f}" width="10" height="10" rx="2"'
                     if arm == "agent" else f'<circle cx="{X(cx):.1f}" cy="{Y(cy):.1f}" r="5"')
            p.append(f'{shape} fill="{c}" stroke="var(--surface-1)" stroke-width="2"><title>{_arm_title(arm)} '
                     f'median: {cx:.0f} ns, error {cy:.3f}, success {s[arm]["success_rate"]:.0%}</title>'
                     f'{"</rect>" if arm == "agent" else "</circle>"}')
            p.append(f'<text x="{X(cx) + 9:.1f}" y="{Y(cy) - 8:.1f}">{_arm_title(arm)}</text>')
    # Legend (identity is never colour alone: lines are fixed plans, points the adaptive arms).
    ly = T + 6
    for m in slots:
        c = f"var({colour_var[m]})"
        mark = (f'<line x1="{W - R + 10}" x2="{W - R + 26}" y1="{ly}" y2="{ly}" stroke="{c}" stroke-width="2"/>'
                if m.startswith("fixed_") else
                f'<rect x="{W - R + 13}" y="{ly - 5}" width="10" height="10" rx="2" fill="{c}"/>' if m == "agent" else
                f'<circle cx="{W - R + 18}" cy="{ly}" r="5" fill="{c}"/>')
        name = f"fixed, {m.split('_')[1]} replica(s)" if m.startswith("fixed_") else _arm_title(m)
        p.append(mark + f'<text x="{W - R + 32}" y="{ly + 4}">{name}</text>')
        ly += 20
    p.append("</svg>")
    return "\n".join(p)


def _plain(x):
    """JSON-safe copy: numpy scalars and arrays become Python values, inf and nan become None."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_outputs(out: dict, out_dir, name: str = "experiment_A") -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{name}{k}" for k in (".json", ".md", "_cost_accuracy.csv", "_cost_accuracy.svg")}
    paths[".json"].write_text(json.dumps(_plain(out), indent=1))
    rows = curve_rows(out)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "method", k))
    with open(paths["_cost_accuracy.csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    paths["_cost_accuracy.svg"].write_text(svg_chart(out))
    paths[".md"].write_text(markdown(out, paths["_cost_accuracy.svg"].name, paths["_cost_accuracy.csv"].name),
                            encoding="utf-8")
    return list(paths.values())


def _len(x) -> str:
    return "not within the pool" if x is None else f"{x:g} ns"


def markdown(out: dict, svg_name: str, csv_name: str) -> str:
    tg, s, v, d = out["targets"], out["summary"], out["verdict"], out["design"]
    zh = {"supported": "假设 A 成立", "not_supported": "假设 A 不成立",
          "reference_inadequate": "参考不够精确，无法判定"}[v["result"]]
    L = [f"# Experiment A: budget allocation for unbiased MD", "",
         f"Written {time.strftime('%Y-%m-%d %H:%M')}. Pool: {out['pool']['source']} — {out['pool']['replicas']} "
         f"replicas of {out['pool']['ns_per_replica']:g} ns; items {', '.join(out['pool']['items'])}; "
         f"{len(out['pool']['held_out'])} replicas held out as the reference.", "",
         f"## Verdict: {v['result']}（{zh}）", ""]
    if v["failed"]:
        L.append("Failed: " + "; ".join(v["failed"]) + ".")
        L.append("")
    a = s["agent"]
    L += [f"- **Reliable?** The agent stopped with an accurate answer in {a['success_rate']:.0%} of "
          f"{a['trials']} trials (95% interval {a['success_interval']}); it stopped with a wrong answer in "
          f"{a['false_stop_rate']:.0%} ({a['false_stop_interval']}); it said INSUFFICIENT_EVIDENCE in "
          f"{(a['correct_insufficient'] + a['over_cautious']) / a['trials']:.0%}. Targets: success ≥ "
          f"{tg['reliability']:.0%}, false stops ≤ {tg['false_stop_max']:.0%}.",
          ]
    for arm, x in v["vs_sequential_ci"].items():
        L.append(f"- **Against {_arm_title(arm)}** (success {x['success_rate']:.0%}): paired saving "
                 f"{x['saving_mean']:.0%}, 95% interval {x['saving_interval']} → "
                 f"{'agent better' if x['agent_better'] else 'agent not better'}"
                 + ("" if x["reliable"] else " (this baseline is not reliable itself)") + ".")
    for n, x in v["vs_fixed"].items():
        L.append(f"- **Against fixed plans of {n} replica(s)** at the same reliability: "
                 + (x["note"] if x["reliable_compute_ns"] is None else
                    f"they need {x['reliable_compute_ns']:.0f} ns; agent saving {x['saving']:.0%}") + ".")
    if v.get("vs_literature_protocols"):
        L.append("- **Protocols people write down in advance** (information, not a condition): " + "; ".join(
            f"{k}: {x['compute_ns']:.0f} ns, accurate in {x['accurate_rate']:.0%}, agent saving "
            f"{x['agent_saving']:.0%}" for k, x in v["vs_literature_protocols"].items()) + ".")
    L += ["", "A fixed plan's reliable cost is read off afterwards from how often it was accurate over the trials. "
          "Nobody can choose that length before running, so an online method with calibrated intervals pays "
          "roughly (its interval factor / the fixed plan's)² more; it is reported, not hidden.", "",
          f"![cost and accuracy]({svg_name})", "",
          f"The numbers behind the figure: `{csv_name}`.", "",
          "## Methods", "",
          "| Method | Success | False stop | Insufficient | Coverage | Compute ns (median, IQR) | Replicas × ns | Looks | C_stop / C_hit |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm in out["arms"]:
        x = s[arm]
        L.append(f"| {_arm_title(arm)} | {x['success_rate']:.0%} | {x['false_stop_rate']:.0%} | "
                 f"{x['correct_insufficient'] + x['over_cautious']} | {x['coverage']:.0%} | "
                 f"{x['compute_ns_median']:.0f} ({x['compute_ns_iqr'][0]:.0f}–{x['compute_ns_iqr'][1]:.0f}) | "
                 f"{x['replicas_median']:g} × {x['ns_per_replica_median']:.0f} | {x['looks_median']:g} | "
                 f"{x['stop_over_hit_median']:.2f} |")
    L += ["", "Success = STOP with an accurate answer. False stop = STOP with a wrong answer (premature "
          "convergence). Insufficient = INSUFFICIENT_EVIDENCE, which is the right answer when the data cannot "
          "support a conclusion. Coverage = the reference lies inside the reported interval. C_hit = the cost at "
          "which the answer first became accurate and stayed so along the method's own path; C_stop / C_hit near 1 "
          "means it stopped as soon as it could have.", "",
          "## Other observables (judged separately)", "",
          "| Method | dG claimed / wrong | Rates claimed / wrong | Landscape claimed / wrong |", "|---|---|---|---|"]
    for arm in out["arms"]:
        c = s[arm]["claims"]
        L.append(f"| {_arm_title(arm)} | " + " | ".join(
            f"{c[o]['claimed']} / {c[o]['claimed_but_wrong']}" for o in ("dg", "rates", "landscape")) + " |")
    L += ["", "A method may stop on the populations alone. Then it must call each other observable converged or "
          "not; a claim that is wrong is counted here.", "", "## Fixed plans", "",
          "| Replicas | Shortest length accurate in ≥ " + f"{tg['reliability']:.0%} of trials | Compute ns | "
          "Median C_hit |", "|---|---|---|---|"]
    for n, f in s["fixed"].items():
        L.append(f"| {n} | {_len(f['reliable_ns'])} | "
                 f"{'-' if f['reliable_compute_ns'] is None else round(f['reliable_compute_ns'])} | "
                 f"{f['c_hit_median']:.0f} |")
    L += ["", "## Reference", ""]
    for k, r in out["reference"].items():
        ok = ", ".join(f"{o} {'yes' if x else 'NO'}" for o, x in r["adequate"].items())
        L.append(f"- {k}: populations {', '.join(f'{n} {p:.3f}' for n, p in r['populations'].items())}; "
                 f"dG {r['dg_kt']:.2f} kT; transitions per replica {r['transitions']}. Precise enough to judge: {ok}."
                 + (f" Known truth: {r['truth']}." if r.get("truth") else ""))
    L += ["", "## Design (fixed before the data)", "",
          f"- Tolerances: populations {tg['population']:g}, dG {tg['dg_kt']:g} kT, rates {tg['rate_rel']:.0%}, "
          f"landscape JS {tg['landscape_js']:g} bits; alpha {tg['alpha']:g}; at least {tg['min_transitions']} "
          "transitions per replica before an occupancy counts.",
          f"- Agent: starts with {d['replicas']} replicas; one planning look at {d['chunk_ns']:g} ns that may not stop; "
          f"then at most {d['agent_looks']} looks that may stop, intervals calibrated for that many (Bonferroni); "
          f"budget {d['budget_ns']:g} ns, at most {d['max_replicas']} replicas of {d['max_ns']:g} ns.",
          f"- Sequential-CI: {', '.join(str(n) for n in d['sequential'])} replicas, a look every {d['chunk_ns']:g} ns, "
          "intervals calibrated for all its looks.",
          "- Fixed plans: " + ", ".join(f"{n} replica(s)" for n in d["families"]) + f", scanned every {d['scan_ns']:g} ns.",
          "- New replicas start from the same prepared structure with new seeds; no method restarts from a chosen "
          "frame. The trials take different replicas out of one pool, so they are not fully independent experiments.",
          ""]
    return "\n".join(L)
