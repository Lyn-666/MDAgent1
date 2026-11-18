"""Step 4: after each round, choose the next action, and check the permissions before doing it.

Plain MD (after agent.decide has judged every item), in this order:
1. Everything is sampled well: stop. Without permission to stop early, go on to the planned budget.
2. An item does not change state at all: switch to enhanced sampling if that is allowed and exists for
   the quantity (agent.decide and agent.run_all do this); otherwise suggest it and go on.
3. Items change state, but their error comes mainly from the spread between replicas: add two
   independent replicas (permission add_replicas, at most Goal.max_replicas). More replicas reduce
   such an error faster than longer ones (Knapp et al. 2018; Grossfield et al. 2018).
4. Items change state but are not precise enough, and the budget is used up: run longer than the
   budget (permission extend, at most Goal.extend_factor times the first budget). Otherwise stop.
5. Replicas sit in different states and never change: suggest restarting from chosen frames (adaptive
   sampling). It is only suggested: segments that start from chosen frames no longer give equilibrium
   populations without further analysis.

Each choice is kept in run.json (notes["actions"]) and decisions.log. An action without permission
becomes a suggestion in the chat.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .messages import post


@dataclass
class Action:
    phase: str          # "plain_md" or "enhanced_sampling"
    action: str         # stop, extend, add_replicas, extend_budget, switch_method, suggest_reseed, ...
    executed: bool
    reason: str
    ns_per_replica: float = 0.0
    time: str = ""


def record(run_dir, state, action: Action, zh: str | None = None, en: str | None = None,
           kind: str = "info", once_key: str | None = None) -> None:
    """Keep the action in run.json and decisions.log; tell the chat if there is a message.

    With once_key, the message is posted only the first time.
    """
    action.time = time.strftime("%Y-%m-%d %H:%M:%S")
    state.notes.setdefault("actions", []).append(asdict(action))
    with open(run_dir / "decisions.log", "a") as f:
        f.write(f"[{action.time}] Action ({action.phase}): {action.action}"
                + ("" if action.executed else " (suggested, not done)") + f". {action.reason}\n\n")
    if zh and (once_key is None or not state.notes.get(once_key)):
        post(run_dir, zh, en or zh, kind)
        if once_key:
            state.notes[once_key] = True
    state.save(run_dir)


def plain_md_next(run_dir, state, reports, decisions, next_ns: float) -> float:
    """Apply rules 1, 3, 4 and 5 to the plain MD decisions. Returns the length of the next round (0: stop).

    May change state.n_replicas (rule 3) and state.loop.max_ns (rule 4).
    """
    goal, ns = state.the_goal(), state.ns_done
    room = round(state.loop.max_ns - ns, 6)
    by_label = {r.label: r for r in reports}
    open_ = [d for d in decisions if d.action not in ("done", "skip")]
    name = run_dir.name

    # Rule 1: everything is sampled well.
    if not open_:
        if room > 0 and not state.allowed("stop_early"):
            record(run_dir, state, Action("plain_md", "extend", True,
                                          "The goal is reached, but there is no permission to stop early.", ns),
                   f"{name}：已经达到目标，但你没有允许提前停，所以按设置跑到每条 {state.loop.max_ns:g} ns。",
                   f"{name}: the goal is reached, but stopping early is not allowed, so it runs to "
                   f"{state.loop.max_ns:g} ns per replica as set.", once_key="said_goal_no_early_stop")
            return min(state.loop.extend_ns, room)
        record(run_dir, state, Action("plain_md", "stop", True, "Every item is sampled well (goal reached).", ns))
        return 0.0

    # Rule 5: replicas sit in different states without any change (suggestion only).
    split = [d.residue for d in open_ if by_label.get(d.residue) is not None
             and by_label[d.residue].verdict == "replicas_disagree" and sum(by_label[d.residue].transitions) == 0]
    if split:
        record(run_dir, state, Action("plain_md", "suggest_reseed", False,
                                      f"Replicas sit in different states without changing: {', '.join(split)}.", ns),
               f"{name}：{', '.join(split)} 在不同重复里停在不同状态，而且从来不变化。普通 MD 在这里估不出比例。"
               "可以考虑从这些状态的帧重新起跑（自适应采样，--sampling adaptive），或者对侧链用增强采样。",
               f"{name}: {', '.join(split)} sit in different states in different replicas and never change, so plain "
               "MD cannot estimate their populations. Consider restarting from frames in these states (adaptive "
               "sampling, --sampling adaptive), or enhanced sampling for side chains.", "suggestion",
               once_key="said_reseed")

    flipping = [d for d in open_ if by_label.get(d.residue) is not None and sum(by_label[d.residue].transitions) > 0]

    # Rule 3: the error comes mainly from the spread between replicas.
    if next_ns > 0 and state.sampling != "adaptive":
        spread = [d.residue for d in flipping if by_label[d.residue].error_from_replicas
                  or by_label[d.residue].verdict == "replicas_disagree"]
        if spread and state.n_replicas < goal.max_replicas:
            add = min(2, goal.max_replicas - state.n_replicas)
            why = (f"The error of {', '.join(spread)} comes mainly from the spread between replicas, so more "
                   "independent replicas help more than longer ones.")
            if state.allowed("add_replicas"):
                state.n_replicas += add
                record(run_dir, state, Action("plain_md", "add_replicas", True,
                                              why + f" Now {state.n_replicas} replicas.", ns),
                       f"{name}：{', '.join(spread)} 的误差主要来自各条重复之间的差别，加更多独立重复比加长更有用。"
                       f"现在加 {add} 条，共 {state.n_replicas} 条；新的重复会先补跑到每条 {ns:g} ns。",
                       f"{name}: the error of {', '.join(spread)} comes mainly from the spread between replicas; more "
                       f"independent replicas help more than longer ones. Adding {add}, now {state.n_replicas}; the new "
                       f"ones first catch up to {ns:g} ns.")
            else:
                record(run_dir, state, Action("plain_md", "add_replicas", False, why, ns),
                       f"{name}：{', '.join(spread)} 的误差主要来自各条重复之间的差别。建议增加重复（需要允许 add_replicas）。",
                       f"{name}: the error of {', '.join(spread)} comes mainly from the spread between replicas. I "
                       "suggest more replicas (needs permission add_replicas).", "suggestion",
                       once_key="said_add_replicas")

    # Rule 4: the budget is used up, items change state but are not precise enough, none is stuck.
    stuck = [d for d in open_ if d.cause == "stalled" or d.stalled]
    if next_ns == 0 and flipping and not stuck:
        first = state.notes.setdefault("first_max_ns", state.loop.max_ns)
        cap = goal.extend_factor * first
        names = ", ".join(d.residue for d in flipping)
        if state.allowed("extend") and ns < cap - 1e-9:
            step = min(state.loop.extend_ns, cap - ns)
            state.loop.max_ns = round(ns + step, 6)
            for d in flipping:
                d.action, d.cause = "extend", ""
            record(run_dir, state, Action("plain_md", "extend_budget", True,
                                          f"Budget used up, but {names} change state and need more data. "
                                          f"New budget {state.loop.max_ns:g} ns (limit {cap:g}).", ns),
                   f"{name}：预算用完了，但 {names} 在状态之间有来回、只是还不够精确。延长到每条 {state.loop.max_ns:g} ns"
                   f"（最多 {cap:g} ns）。",
                   f"{name}: the budget is used up, but {names} change state and only need more data. Extending to "
                   f"{state.loop.max_ns:g} ns per replica (at most {cap:g}).")
            return step
        if not state.allowed("extend"):
            record(run_dir, state, Action("plain_md", "extend_budget", False,
                                          f"Budget used up; {names} need more data.", ns))
    return next_ns
