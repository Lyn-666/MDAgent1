"""How the agent spends simulation time: how many replicas, how long, and when to stop.

This is the agent's decision rule. The live run loop (agent.py) and the offline benchmark
(experiment_a.py) both call `next_step`, so the benchmark measures the agent that users run.

After each look at the data the agent does one of three things:

- STOP: every item is precise enough (the caller says so, from its own calibrated intervals) and
  sampling is adequate. The first look only plans and may not stop, so it spends none of alpha;
  after it, at most `stop_looks` looks may stop, and the caller's intervals are calibrated for
  exactly that many (Bonferroni). Fewer looks allow narrower intervals, which is why the agent
  jumps straight to the length it projects instead of looking every few ns.
- INSUFFICIENT_EVIDENCE: the data cannot support a conclusion within the limits: the budget or the
  looks are used up, an item never changed state after `patience_ns`, or the projected cost is far
  beyond the budget. Saying so is a correct answer, not a failure to hide.
- continue, with a new number of replicas and a new length per replica (`allocate`): the cheapest
  allocation projected to reach the precision. An error is split into the part inside replicas,
  which falls with all the data together, and the extra spread between replicas, which only more
  replicas reduce; that extra part counts only when it is larger than chance (chi-square test).
  Every new replica pays its equilibration, so running longer is preferred unless the spread
  between replicas makes it useless. New replicas start from the prepared structure with new seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

STOP, INSUFFICIENT, CONTINUE = "STOP", "INSUFFICIENT_EVIDENCE", "continue"


@dataclass
class ItemState:
    """What the agent knows about one item (a residue, a distance, a turn) after a look."""

    label: str
    se: dict               # state -> standard error of its population
    within: float          # error part inside the replicas (worst state)
    between: float         # spread between replicas (worst state)
    transitions: list      # state changes per replica
    converged: bool        # precise enough by the caller's calibrated interval, and adequately sampled
    cut_ns: float = 0.0    # left out at the start of each replica (automated burn-in); the errors come from the rest


@dataclass
class Limits:
    max_replicas: int
    max_ns: float          # per replica
    budget_ns: float       # production ns, all replicas together
    chunk_ns: float = 5.0
    patience_ns: float = 0.0     # Before 2026-11-22: 20 ns. The budget rule below needs no fixed waiting time.
    give_up_factor: float = 3.0
    margin: float = 1.2
    final_margin: float = 1.6
    stop_looks: int = 3
    min_transitions: int = 4
    # Student's t (n - 1 degrees of freedom) instead of z where the replica spread is real. True when z is the
    # factor of a confidence interval; False for the plain standard-error criterion (z = 1 is not an interval).
    t_dist: bool = False
    # Level of the chi-square test above which the spread between replicas counts in the plan (it then calls for
    # more replicas). Below the verdict's 0.95 on purpose: with 3 replicas the test has little power, and a spread
    # that is only suggestive is cheaper to resolve by adding replicas early than by running long and failing.
    # Chosen on the synthetic development families (review revision, 2026-11-23). The verdict's t factor still
    # needs the 0.95 test.
    spread_level: float = 0.8


@dataclass
class Step:
    action: str            # STOP, INSUFFICIENT or CONTINUE
    replicas: int          # for CONTINUE: replicas in the next round (new ones catch up first)
    ns: float              # for CONTINUE: total length per replica after the next round
    reason: str
    planning: bool = False
    # Why, as a short code for messages in other languages: converged, looks_used, budget_used, stuck,
    # hopeless, too_few_changes, projected.
    kind: str = ""
    # Items left out because plain MD cannot answer them within the limits (label -> "stuck" or "hopeless").
    # They are reported as INSUFFICIENT_EVIDENCE on their own; the other items go on to their precision.
    set_aside: dict = field(default_factory=dict)
    # Whether this look could have stopped, so it counts against stop_looks. A look at which some item still
    # has too few state changes cannot stop (STOP needs adequate sampling), so it spends none of alpha.
    counted: bool = False


def z_bonferroni(alpha: float, looks: int) -> float:
    """Two-sided interval factor with alpha shared over a known largest number of looks."""
    return NormalDist().inv_cdf(1 - alpha / (2 * max(1, looks)))


def chi2_quantile(p: float, k: int) -> float:
    """Wilson-Hilferty approximation of the chi-square quantile (good to a few percent for k >= 2)."""
    zp = NormalDist().inv_cdf(p)
    return k * (1 - 2 / (9 * k) + zp * math.sqrt(2 / (9 * k))) ** 3


def allocate(items: list, n: int, ns: float, z: float, tolerance: float, max_n: int, max_ns: float,
             equil_ns: float, t_dist: bool = False, spread_level: float = 0.95):
    """The cheapest (replicas, ns per replica) projected to reach the precision, or None.

    `items` are ItemState (or dicts with "within" and "between"). Lengths cannot shrink.
    t_dist: where the spread between replicas is real (uncertainty.replica_spread_real, the same test as the
    verdict in basins.judge and diagnose.judge), the interval of m replicas uses Student's t with m - 1
    degrees of freedom (uncertainty.t_factor), so more replicas also narrow the factor itself.
    spread_level: the spread between replicas counts as extra error, which only more replicas reduce, when the
    chi-square test passes at this level (0.95: significant; lower: also when only suggestive).
    Returns (replicas, ns per replica, projected compute ns).
    """
    from .uncertainty import t_factor

    def part(i, k, default=None):
        return i.get(k, default) if isinstance(i, dict) else getattr(i, k, default)

    from .uncertainty import replica_spread_real

    best = None
    for m in range(n, max_n + 1):
        need = 0.0
        for i in items:
            w, b = part(i, "within"), part(i, "between")
            real = replica_spread_real(w, b, n)                     # the verdict's test: t
            seen = real or replica_spread_real(w, b, n, spread_level)
            extra = (b * b - w * w) * n / m if seen else 0.0     # shrinks with more replicas only
            zm = t_factor(z, m - 1) if (t_dist and real) else z
            room = (tolerance / zm) ** 2 - extra
            if room <= 0:
                need = float("inf")
                break
            # The inside part was measured on the kept length (ns - cut); a burn-in is a transient of fixed duration,
            # so a longer replica keeps (length - cut).
            cut = min(part(i, "cut_ns", 0.0) or 0.0, ns)
            need = max(need, (ns - cut) * (w * w * n / m) / room + cut)   # length at which the inside part fits
        length = max(ns, need)
        if length > max_ns + 1e-9:
            continue
        cost = m * (length + equil_ns)
        if best is None or cost < best[2]:
            best = (m, length, cost)
    return best


# Raised when the rule changes, so a resumed run decides again under the new rule.
# 5 (2026-11-23): Student's t where the replica spread is real; automated burn-in in projection runs.
# 6 (2026-11-23): a spread between replicas that is only suggestive (chi-square at 0.8) already calls for replicas;
#   give up only when even the optimistic spread (95 % lower bound) cannot reach the precision, and otherwise add
#   replicas to measure the spread.
# 7 (2026-11-23): when everything that changed state is precise enough, items that never changed stop plain MD.
# 8 (2026-11-24): automated burn-in counts a stretch without changes as one sample (uncertainty.indicator_samples),
#   so it no longer cuts away the part with the changes.
# 9 (2026-11-24): structure stability stops only when the departure-rate bound reaches its goal (default 10 per µs).
RULE_VERSION = 9


def stuck_projection(n: int, ns: float, z: float, tolerance: float) -> tuple[float, float, float]:
    """For an item that never changed state: how much data would be needed at best.

    Zero changes in A = n x ns of data: at 95% confidence the rate is at most 3/A (Poisson). To resolve even a
    small basin (population about the tolerance) to within the tolerance, the error floor (about
    2 q (1 - q) / sqrt(N) after N changes) needs N >= (2 (1 - tolerance) z)^2 changes. At the fastest rate that
    takes N A / 3 of data. Returns (fastest rate per ns, changes needed, data needed in ns).
    """
    seen = max(n * ns, 1e-9)
    rate = 3.0 / seen
    need = (2 * (1 - tolerance) * z) ** 2
    return rate, need, need / rate


def optimistic(item, n: int) -> dict:
    """The item with its spread between replicas at the 95 % lower confidence bound (chi-square, n - 1 degrees of
    freedom). With few replicas the spread itself is a rough number (3 replicas: 2 degrees of freedom), so a
    projection from it can look hopeless by chance. Giving up needs even this optimistic projection to fail, as
    for an item that never changed state (stuck_projection uses the fastest rate the data allow)."""
    def part(k):
        return item[k] if isinstance(item, dict) else getattr(item, k)
    w, b = part("within"), part("between")
    cut = (item.get("cut_ns", 0.0) if isinstance(item, dict) else getattr(item, "cut_ns", 0.0)) or 0.0
    if n < 2:
        return {"within": w, "between": b, "cut_ns": cut}
    k = n - 1
    return {"within": w, "between": b * math.sqrt(k / chi2_quantile(0.95, k)), "cut_ns": cut}


def _next_step(items: list, n: int, ns: float, look: int, tests_before: int, z: float, tolerance: float,
              limits: Limits, equil_ns: float = 0.6) -> Step:
    """One decision after look number `look` (1 = the first). `tests_before`: earlier looks that could stop.

    An item that plain MD cannot answer within the limits (it never changed state after patience_ns, or its
    own projected cost is far beyond the budget) is set aside: it gets INSUFFICIENT_EVIDENCE on its own, and
    the time goes to the items that can still be answered. Only when nothing answerable is left does the
    whole run stop with INSUFFICIENT_EVIDENCE.
    """
    planning = look == 1
    max_n, max_ns = limits.max_replicas, limits.max_ns
    spent = n * ns
    aside = {}
    if ns >= (limits.patience_ns or 0.0):
        # An item that never changed state is set aside only when even the fastest rate consistent with that
        # cannot give enough changes within the rest of the budget (uncertainty.never_changing_hopeless, the same
        # rule as the rounds mode). More budget: more patience.
        from .uncertainty import changes_needed, never_changing_hopeless

        need = changes_needed(n, limits.min_transitions, tolerance, z)
        # Exposure without a change: only the kept time (a change inside a cut burn-in is not seen).
        cut = max([i.cut_ns for i in items if sum(i.transitions) == 0] or [0.0])
        if never_changing_hopeless(n * max(ns - cut, 0.0), limits.budget_ns - spent, need):
            aside = {i.label: "stuck" for i in items if sum(i.transitions) == 0}
    if spent >= limits.budget_ns / 4:
        # A projection from very little data is too noisy to give up on: wait for a quarter of the budget.
        for i in items:
            if i.label in aside or i.converged or not (i.transitions and min(i.transitions) >= limits.min_transitions):
                continue
            own = allocate([i], n, ns, z, tolerance, max_n, max_ns, equil_ns, limits.t_dist, limits.spread_level)
            if own is None or own[2] > limits.give_up_factor * limits.budget_ns:
                best = allocate([optimistic(i, n)], n, ns, z, tolerance, max_n, max_ns, equil_ns, limits.t_dist,
                                limits.spread_level)
                if best is None or best[2] > limits.give_up_factor * limits.budget_ns:
                    aside[i.label] = "hopeless"   # even the optimistic spread cannot reach the precision
    active = [i for i in items if i.label not in aside]
    # Only a look that could have stopped counts as a test: not the planning look, and not a look at which an
    # item still has too few state changes (STOP needs adequate sampling, so no false stop can happen there).
    could_stop = (not planning and bool(active)
                  and all(i.transitions and min(i.transitions) >= limits.min_transitions for i in active))
    tests = tests_before + (1 if could_stop else 0)
    left_out = (f" Left out, plain MD cannot answer them within the limits: {', '.join(aside)}."
                if aside and active else "")
    if not active:
        stuck = [k for k, v in aside.items() if v == "stuck"]
        if stuck:
            rate, need, data = stuck_projection(n, ns, z, tolerance)
            rest = max(limits.budget_ns - n * ns, 0.0)
            why = (f"{', '.join(stuck)} never changed state in {n * ns:g} ns of all replicas together; even at the "
                   f"fastest rate that allows (one change per {1 / rate:.3g} ns), the precision needs about {need:.0f} "
                   f"changes, that is {data:.3g} ns more, more than the {rest:.3g} ns left of the budget of "
                   f"{limits.budget_ns:g} ns")
        else:
            why = "the projected cost is far beyond the budget"
        return Step(INSUFFICIENT, n, ns, f"{why}; plain MD cannot answer within the limits", planning,
                    "stuck" if "stuck" in aside.values() else "hopeless", aside)
    # Everything that changed state is done; only items that never changed state are left (the user's decision,
    # 2026-11-23). With a large budget the never-changing rule would wait long (about 3 / 21 of the budget), and
    # the whole run with it. Stop plain MD now: those items get INSUFFICIENT_EVIDENCE on their own and enhanced
    # sampling is proposed for them (or runs, with permission switch_method).
    never = [i for i in active if sum(i.transitions) == 0]
    rest = [i for i in active if sum(i.transitions) > 0]
    if not planning and never and rest and all(i.converged for i in rest):
        for i in never:
            aside[i.label] = "stuck"
        return Step(STOP, n, ns, "every item that changed state is precise enough; "
                    f"{', '.join(i.label for i in never)} never changed state in {n * ns:g} ns of all replicas "
                    "together, so plain MD stops instead of waiting for them; enhanced sampling can sample them",
                    planning, "others_done", aside)
    if not planning and all(i.converged for i in active):
        return Step(STOP, n, ns, "every item that plain MD can answer is precise enough and adequately sampled"
                    + left_out, planning, "converged", aside)
    if tests >= limits.stop_looks:
        return Step(INSUFFICIENT, n, ns, "the largest number of looks is reached before the precision" + left_out,
                    planning, "looks_used", aside)
    if spent >= limits.budget_ns - 1e-9 or (ns >= max_ns - 1e-9 and n >= max_n):
        return Step(INSUFFICIENT, n, ns, "the budget is used up before the precision" + left_out, planning,
                    "budget_used", aside)
    last = tests == limits.stop_looks - 1          # the next look is the last one that may stop
    margin = limits.final_margin if last else limits.margin
    adequate = all(i.transitions and min(i.transitions) >= limits.min_transitions for i in active)
    new_n, new_ns = n, ns
    if not adequate:
        new_ns = min(max_ns, (4 if last else 2) * ns)
        short = [f"{i.label} {list(i.transitions)}" for i in active
                 if not (i.transitions and min(i.transitions) >= limits.min_transitions)]
        why = (f"too few state changes to trust the errors yet (changes per replica: {'; '.join(short)}; every "
               f"replica needs at least {limits.min_transitions}); running longer")
    else:
        plan = allocate(active, n, ns, z, tolerance, max_n, max_ns, equil_ns, limits.t_dist, limits.spread_level)
        hopeless = plan is None or plan[2] > limits.give_up_factor * limits.budget_ns
        hope = None
        if hopeless:
            hope = allocate([optimistic(i, n) for i in active], n, ns, z, tolerance, max_n, max_ns, equil_ns,
                            limits.t_dist, limits.spread_level)
            if hope is not None and hope[2] > limits.give_up_factor * limits.budget_ns:
                hope = None
        # The spread between few replicas is uncertain: the point projection fails, but the optimistic one does not.
        # Measure the spread with more replicas first (the user's choice of 2026-11-23: a spread that is only
        # suggestive already calls for replicas). New replicas catch up to the current length, and every replica
        # runs one chunk more (the live loop runs new replicas only within a round of positive length); the
        # budget holds. Not possible at the length limit.
        # One small step (at most a tenth of the current length): enough for the new replicas to run in the live loop,
        # without paying a full round on every replica (with a 2 us limit a round is 100 ns).
        longer = min(max_ns, ns + min(limits.chunk_ns, max(0.1 * ns, 1e-3)))
        explore = min(max_n, int((limits.budget_ns + 1e-9) // max(longer, 1e-9)))
        if hopeless and hope is not None and explore > n and longer > ns + 1e-9:
            return Step(CONTINUE, explore, round(longer, 6),
                        f"the spread between {n} replicas is too uncertain to project from; adding replicas to "
                        f"measure it ({explore} replicas of {longer:g} ns)" + left_out, planning, "explore_replicas",
                        aside)
        if hopeless and hope is not None:
            plan = hope                                   # no room for more replicas: go on as the optimistic plan says
        if hopeless and hope is None:
            if spent < limits.budget_ns / 4:
                plan = (n, min(max_ns, max(2 * ns, limits.budget_ns / 4 / n)), 0.0)
            else:
                need = "no allocation within the limits reaches it" if plan is None else \
                    f"the projected cost is {plan[2]:.0f} ns"
                return Step(INSUFFICIENT, n, ns, f"{need}, more than {limits.give_up_factor:g} times the budget "
                            f"({limits.budget_ns:g} ns), so more plain MD will not reach the precision" + left_out,
                            planning, "hopeless", aside)
        new_n, new_ns = plan[0], min(max_ns, plan[1] * margin)
        why = (f"projected: {new_n} replicas of {plan[1]:.3g} ns reach the precision for the least compute "
               f"(plus a {margin - 1:.0%} margin)")
    if new_n * new_ns > limits.budget_ns:            # stay inside the budget
        new_ns = max(ns, limits.budget_ns / new_n)
        if new_n * new_ns > limits.budget_ns + 1e-9:
            new_n, new_ns = n, min(max_ns, limits.budget_ns / n)
    if new_n == n and new_ns <= ns + 1e-9:
        new_ns = min(max_ns, ns + limits.chunk_ns)
    return Step(CONTINUE, new_n, round(new_ns, 6), why + left_out, planning,
                "projected" if adequate else "too_few_changes", aside)


def next_step(items: list, n: int, ns: float, look: int, tests_before: int, z: float, tolerance: float,
              limits: Limits, equil_ns: float = 0.6) -> Step:
    """See _next_step. Also says whether the look counted against stop_looks (Step.counted)."""
    step = _next_step(items, n, ns, look, tests_before, z, tolerance, limits, equil_ns)
    aside = set(step.set_aside)
    active = [i for i in items if i.label not in aside]
    step.counted = (look != 1 and bool(active)
                    and all(i.transitions and min(i.transitions) >= limits.min_transitions for i in active))
    return step
