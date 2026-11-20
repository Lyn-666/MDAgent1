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


@dataclass
class Limits:
    max_replicas: int
    max_ns: float          # per replica
    budget_ns: float       # production ns, all replicas together
    chunk_ns: float = 5.0
    patience_ns: float = 20.0
    give_up_factor: float = 3.0
    margin: float = 1.2
    final_margin: float = 1.6
    stop_looks: int = 3
    min_transitions: int = 4


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


def z_bonferroni(alpha: float, looks: int) -> float:
    """Two-sided interval factor with alpha shared over a known largest number of looks."""
    return NormalDist().inv_cdf(1 - alpha / (2 * max(1, looks)))


def chi2_quantile(p: float, k: int) -> float:
    """Wilson-Hilferty approximation of the chi-square quantile (good to a few percent for k >= 2)."""
    zp = NormalDist().inv_cdf(p)
    return k * (1 - 2 / (9 * k) + zp * math.sqrt(2 / (9 * k))) ** 3


def allocate(items: list, n: int, ns: float, z: float, tolerance: float, max_n: int, max_ns: float,
             equil_ns: float):
    """The cheapest (replicas, ns per replica) projected to reach the precision, or None.

    `items` are ItemState (or dicts with "within" and "between"). Lengths cannot shrink.
    Returns (replicas, ns per replica, projected compute ns).
    """
    def part(i, k):
        return i[k] if isinstance(i, dict) else getattr(i, k)

    best = None
    limit = chi2_quantile(0.95, n - 1) / (n - 1) if n > 1 else float("inf")
    for m in range(n, max_n + 1):
        need = 0.0
        for i in items:
            w, b = part(i, "within"), part(i, "between")
            real = w > 0 and (b * b) / (w * w) > limit
            extra = (b * b - w * w) * n / m if real else 0.0     # shrinks with more replicas only
            room = (tolerance / z) ** 2 - extra
            if room <= 0:
                need = float("inf")
                break
            need = max(need, ns * (w * w * n / m) / room)        # length at which the inside part fits
        length = max(ns, need)
        if length > max_ns + 1e-9:
            continue
        cost = m * (length + equil_ns)
        if best is None or cost < best[2]:
            best = (m, length, cost)
    return best


def next_step(items: list, n: int, ns: float, look: int, tests_before: int, z: float, tolerance: float,
              limits: Limits, equil_ns: float = 0.6) -> Step:
    """One decision after look number `look` (1 = the first). `tests_before`: earlier looks that could stop.

    An item that plain MD cannot answer within the limits (it never changed state after patience_ns, or its
    own projected cost is far beyond the budget) is set aside: it gets INSUFFICIENT_EVIDENCE on its own, and
    the time goes to the items that can still be answered. Only when nothing answerable is left does the
    whole run stop with INSUFFICIENT_EVIDENCE.
    """
    planning = look == 1
    tests = tests_before + (0 if planning else 1)
    max_n, max_ns = limits.max_replicas, limits.max_ns
    spent = n * ns
    aside = {}
    if ns >= limits.patience_ns:
        aside = {i.label: "stuck" for i in items if sum(i.transitions) == 0}
    if spent >= limits.budget_ns / 4:
        # A projection from very little data is too noisy to give up on: wait for a quarter of the budget.
        for i in items:
            if i.label in aside or i.converged or not (i.transitions and min(i.transitions) >= limits.min_transitions):
                continue
            own = allocate([i], n, ns, z, tolerance, max_n, max_ns, equil_ns)
            if own is None or own[2] > limits.give_up_factor * limits.budget_ns:
                aside[i.label] = "hopeless"
    active = [i for i in items if i.label not in aside]
    left_out = (f" Left out, plain MD cannot answer them within the limits: {', '.join(aside)}."
                if aside and active else "")
    if not active:
        why = ("; ".join(f"{k} never changed state in {ns:g} ns per replica" for k, v in aside.items() if v == "stuck")
               or "the projected cost is far beyond the budget")
        return Step(INSUFFICIENT, n, ns, f"{why}; plain MD cannot answer within the limits", planning,
                    "stuck" if "stuck" in aside.values() else "hopeless", aside)
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
        why = "too few state changes to trust the errors yet; running longer"
    else:
        plan = allocate(active, n, ns, z, tolerance, max_n, max_ns, equil_ns)
        if plan is None or plan[2] > limits.give_up_factor * limits.budget_ns:
            if spent < limits.budget_ns / 4:
                plan = (n, min(max_ns, max(2 * ns, limits.budget_ns / 4 / n)), 0.0)
            else:
                need = "no allocation within the limits reaches it" if plan is None else \
                    f"the projected cost is {plan[2]:.0f} ns"
                return Step(INSUFFICIENT, n, ns, f"{need}, more than {limits.give_up_factor:g} times the budget "
                            f"({limits.budget_ns:g} ns), so more plain MD will not reach the precision" + left_out,
                            planning, "hopeless", aside)
        new_n, new_ns = plan[0], min(max_ns, plan[1] * margin)
        why = (f"projected: {new_n} replicas of {plan[1]:.1f} ns reach the precision for the least compute "
               f"(plus a {margin - 1:.0%} margin)")
    if new_n * new_ns > limits.budget_ns:            # stay inside the budget
        new_ns = max(ns, limits.budget_ns / new_n)
        if new_n * new_ns > limits.budget_ns + 1e-9:
            new_n, new_ns = n, min(max_ns, limits.budget_ns / n)
    if new_n == n and new_ns <= ns + 1e-9:
        new_ns = min(max_ns, ns + limits.chunk_ns)
    return Step(CONTINUE, new_n, round(new_ns, 6), why + left_out, planning,
                "projected" if adequate else "too_few_changes", aside)
