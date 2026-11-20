"""Step 4: after each round, choose the next action, and check the permissions before doing it.

Plain MD (after agent.decide has judged every item), in this order:
1. Everything is sampled well: stop. Without permission to stop early, go on to the planned budget.
2. An item does not change state at all: switch to enhanced sampling if that is allowed and exists for
   the quantity (agent.decide and agent.run_all do this); otherwise suggest it and go on.
3. Items change state, but the replicas clearly disagree (populations differ by more than three
   combined errors): add two independent replicas (permission add_replicas, at most Goal.max_replicas),
   but only in the first half of the budget, because new replicas must first run all earlier rounds.
   More replicas help such an error more than longer ones (Knapp et al. 2018; Grossfield et al. 2018).
   A plain "the spread between replicas is larger than the error inside them" is not enough: with three
   replicas that estimate is too noisy, and it made the agent add replicas every round.
4. Items change state but are not precise enough, and the budget is used up: run longer than the
   budget (permission extend, at most Goal.extend_factor times the first budget). Otherwise stop.
5. Replicas sit in different states and never change: suggest restarting from chosen frames (adaptive
   sampling). It is only suggested: segments that start from chosen frames no longer give equilibrium
   populations without further analysis.
6. No gain within the limit: after at least two rounds and a quarter of the budget, estimate for each open
   item how long it would need: state changes come at the rate seen so far (the mean over replicas, which
   is optimistic), and the error falls as one over the square root of the time. If even this estimate is
   more than NO_GAIN_MARGIN times the user's limit for every open item, more plain MD would not reach the
   goal in time: stop early (permission stop_early) with the reason "not_reachable", and suggest a higher
   limit or another method. Without the permission, only say so.

Since 2026-11-20 the default (auto) permissions are only stop_early: the user's time and replica numbers
are limits, and another method needs the user's yes. Rules 3 and 4 then become suggestions that say how
to confirm them.

Each choice is kept in run.json (notes["actions"]) and decisions.log. An action without permission
becomes a suggestion in the chat.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .messages import post

NO_GAIN_MARGIN = 1.5   # Rule 6 stops only when even the optimistic estimate is this much over the limit.


def projected_ns(report, ns: float, precision: float | None, min_flips: int) -> float | None:
    """Rough total ns per replica an item needs to meet the goal, from what was seen so far.

    None when it cannot be estimated (no state change yet). Optimistic on purpose: it uses the mean
    rate of state changes over replicas, so a slow replica does not make the agent give up too early.
    """
    if report is None or ns <= 0 or not report.transitions or sum(report.transitions) == 0:
        return None
    need = ns
    mean_changes = sum(report.transitions) / len(report.transitions)
    if min(report.transitions) < min_flips:
        need = max(need, ns * min_flips / mean_changes)
    if precision and report.standard_error:
        err = max(report.standard_error.values())
        if err > precision:
            need = max(need, ns * (err / precision) ** 2)
    return need


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
    if next_ns > 0 and state.sampling != "adaptive" and ns < 0.5 * state.loop.max_ns:
        spread = [d.residue for d in flipping if by_label[d.residue].replicas_differ
                  or by_label[d.residue].verdict == "replicas_disagree"]
        if spread and state.n_replicas < goal.max_replicas:
            add = min(2, goal.max_replicas - state.n_replicas)
            why = (f"The replicas clearly disagree about {', '.join(spread)}, so more independent replicas help "
                   "more than longer ones.")
            if state.allowed("add_replicas"):
                state.n_replicas += add
                record(run_dir, state, Action("plain_md", "add_replicas", True,
                                              why + f" Now {state.n_replicas} replicas.", ns),
                       f"{name}：各条重复对 {', '.join(spread)} 的结果明显不一致，加更多独立重复比加长更有用。"
                       f"现在加 {add} 条，共 {state.n_replicas} 条；新的重复会先补跑到每条 {ns:g} ns。",
                       f"{name}: the replicas clearly disagree about {', '.join(spread)}; more "
                       f"independent replicas help more than longer ones. Adding {add}, now {state.n_replicas}; the new "
                       f"ones first catch up to {ns:g} ns.")
            else:
                record(run_dir, state, Action("plain_md", "add_replicas", False, why, ns),
                       f"{name}：各条重复对 {', '.join(spread)} 的结果明显不一致，增加独立重复比加长更有用。"
                       f"你定的是 {state.n_replicas} 条重复，我不会自己加。如果同意，可以说“给 {name} 加 2 条重复”。",
                       f"{name}: the replicas clearly disagree about {', '.join(spread)}; more independent replicas "
                       f"help more than longer ones. You set {state.n_replicas} replicas, so I do not add any myself. "
                       f"If you agree, say \"add 2 replicas to {name}\".", "suggestion",
                       once_key="said_add_replicas")

    # Rule 6: no gain within the user's limit.
    going_on = [d for d in open_ if d.action == "extend"]
    # Not for a change to see (event_seen): when a rare event comes cannot be estimated from the rate so far.
    if next_ns > 0 and going_on and state.quantity != "event_seen" and len(state.rounds_ns) >= 2 \
            and ns >= 0.25 * state.loop.max_ns - 1e-9:
        needs = {}
        for d in going_on:
            need = projected_ns(by_label.get(d.residue), ns, goal.precision, state.loop.min_flips_per_replica)
            if need is None or need <= NO_GAIN_MARGIN * state.loop.max_ns:
                needs = {}
                break
            needs[d.residue] = need
        if needs:
            longest = max(needs.values())
            items_zh = "，".join(f"{k} 约 {v:.0f} ns" for k, v in needs.items())
            items_en = ", ".join(f"{k} about {v:.0f} ns" for k, v in needs.items())
            what = (f"按目前的速度，要达到目标（每条重复至少来回 {state.loop.min_flips_per_replica} 次、"
                    f"误差不超过 {goal.precision:g}），每条重复大约需要：{items_zh}。这超过了你定的上限 "
                    f"{state.loop.max_ns:g} ns，所以在上限内再跑下去也达不到。")
            what_en = (f"At the rate so far, reaching the goal (at least {state.loop.min_flips_per_replica} state "
                       f"changes per replica and errors of at most {goal.precision:g}) needs per replica: "
                       f"{items_en}. That is more than your limit of {state.loop.max_ns:g} ns, so more of the same "
                       "will not reach it within the limit.")
            state.notes["projected_ns"] = {k: round(v, 1) for k, v in needs.items()}
            if state.allowed("stop_early"):
                for d in going_on:
                    d.action = "enhanced_sampling" if state.quantity == "chi1_populations" else "stalled"
                    d.cause = "not_reachable"
                record(run_dir, state, Action("plain_md", "stop", True,
                                              f"No gain within the limit: needs {items_en} (limit "
                                              f"{state.loop.max_ns:g} ns).", ns),
                       f"{name}：{what}普通 MD 在每条 {ns:g} ns 处提前停下，省下剩余的时间。如果仍想用普通 MD，"
                       f"可以把上限提高到约 {longest:.0f} ns（说“把 {name} 加长到 {longest:.0f} ns”）。",
                       f"{name}: {what_en} Plain MD stops early at {ns:g} ns per replica to save the rest. To stay "
                       f"with plain MD, raise the limit to about {longest:.0f} ns (say \"run {name} to "
                       f"{longest:.0f} ns\").", "info")
                return 0.0
            record(run_dir, state, Action("plain_md", "stop", False, f"No gain within the limit: needs {items_en}.", ns),
                   f"{name}：{what}运行会按你的设置继续到上限。",
                   f"{name}: {what_en} The run goes on to the limit as you set it.", "suggestion",
                   once_key="said_no_gain")

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
            need = (ns + state.loop.extend_ns if state.quantity == "event_seen" else
                    max((projected_ns(by_label.get(d.residue), ns, goal.precision, state.loop.min_flips_per_replica)
                         or ns) for d in flipping))
            target = max(round(need + 0.5), round(ns + state.loop.extend_ns))
            if state.quantity == "event_seen":
                zh = (f"{name}：你定的上限每条 {ns:g} ns 用完了，还没看到 {names} 的变化（它们在状态之间有来回，"
                      f"但没有在目标状态停够时间）。我不会超过你定的上限。如果同意加长，可以说“把 {name} 加长到 "
                      f"{target} ns”；也可以用自适应采样，从最接近目标的帧重新起跑。")
                en = (f"{name}: your limit of {ns:g} ns per replica is used up and the change of {names} was not seen "
                      "(they change state, but did not stay in the target state long enough). I do not go past your "
                      f"limit. If you agree to run longer, say \"run {name} to {target} ns\"; or use adaptive "
                      "sampling to restart from the frames closest to the target.")
            else:
                zh = (f"{name}：你定的上限每条 {ns:g} ns 用完了。{names} 在状态之间有来回，只是数据还不够；按目前的速度，"
                      f"大约需要每条 {need:.0f} ns。我不会超过你定的上限。如果同意加长，可以说“把 {name} 加长到 {target} ns”。")
                en = (f"{name}: your limit of {ns:g} ns per replica is used up. {names} change state but need more "
                      f"data; at the rate so far, about {need:.0f} ns per replica. I do not go past your limit. If you "
                      f"agree to run longer, say \"run {name} to {target} ns\".")
            record(run_dir, state, Action("plain_md", "extend_budget", False,
                                          f"Budget used up; {names} need more data (about {need:.0f} ns).", ns),
                   zh, en, "suggestion", once_key="said_extend")
    return next_ns
