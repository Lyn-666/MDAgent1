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
    if report.total_ns_override is not None:
        # Adaptive sampling: transitions are counted per segment and judged in total (diagnose.judge_pooled).
        if report.verdict == "rare_flips" and sum(report.transitions) > 0:
            need = max(need, ns * 2)   # Needs more segments; how many cannot be told from a pooled count.
    else:
        mean_changes = sum(report.transitions) / len(report.transitions)
        if min(report.transitions) < min_flips:
            need = max(need, ns * min_flips / mean_changes)
    if precision and report.standard_error:
        err = max(report.standard_error.values())
        if err > precision:
            need = max(need, ns * (err / precision) ** 2)
    return need


def spread_labels(reports) -> list[str]:
    """Items whose replicas clearly disagree, so more independent replicas help more than longer ones.

    Shared with the budget comparison (budget.py), so that the replayed policy is the real one.
    """
    return [r.label for r in reports if r.replicas_differ or r.verdict == "replicas_disagree"]


def stalled_labels(reports, ns_done: float, patience_ns: float) -> list[str]:
    """Items that have not changed state once, after long enough that plain MD is unlikely to show it."""
    return [r.label for r in reports if sum(r.transitions) == 0 and ns_done >= patience_ns]


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
        spread = spread_labels([by_label[d.residue] for d in flipping])
        if spread and state.n_replicas < goal.max_replicas:
            add = min(2, goal.max_replicas - state.n_replicas)
            why = (f"The replicas clearly disagree about {', '.join(spread)}, so more independent replicas help "
                   "more than longer ones.")
            if state.allowed("add_replicas"):
                state.notes.setdefault("replicas_at_start", state.n_replicas)
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
                    from .agent import BIASABLE
                    d.action = "enhanced_sampling" if state.quantity in BIASABLE else "stalled"
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


STEP_ZH = {
    "converged": "所有项目都已足够精确，而且采样充分",
    "looks_used": "可用的查看次数用完了，还没达到精度",
    "budget_used": "预算用完了，还没达到精度",
    "stuck": "有项目一直没有在状态之间变化，普通 MD 回答不了",
    "hopeless": "按投影，达到精度需要的计算量远超预算，再跑普通 MD 也达不到",
    "too_few_changes": "状态变化还太少，误差还不可信，先跑长一些",
    "projected": "按投影，这是达到精度花费最少的分配",
    "explore_replicas": "重复之间的差别只靠几条重复估计，还测不准；先加重复把它测准，再决定要跑多长",
}


def recount_tests(actions: list) -> int:
    """Looks that could have stopped, from the recorded plain MD looks: not the planning look, and not a
    look that only ran longer because the state changes were too few."""
    n = 0
    for a in actions:
        reason = a.get("reason", "")
        if a.get("phase") != "plain_md" or not reason.startswith("Look ") or "(planning only)" in reason:
            continue
        if "too few state changes" not in reason:
            n += 1
    return n


def allocation_next(run_dir, state, reports, decisions) -> float:
    """Projection allocation (allocation.py) in the live loop. Returns the length of the next round (0: stop).

    May raise state.n_replicas (new replicas first catch up, see agent.run_loop). Items that decide() has
    already handed to enhanced sampling are left out when they will really get it. Every choice is kept in run.json (notes["actions"],
    notes["allocation"]) and told in the chat with its reason.
    """
    from .allocation import CONTINUE, INSUFFICIENT, STOP, ItemState, Limits, next_step

    goal, ns, n, name = state.the_goal(), state.ns_done, state.n_replicas, run_dir.name
    by = {r.label: r for r in reports}
    # Items decide() handed to enhanced sampling are left out only when they will really get it (permission, and
    # enhanced sampling not switched off); otherwise this rule judges them like the others (they may be set aside).
    handed = state.can_switch() and state.enhanced.get("enabled", True)
    live = [d for d in decisions if d.action != "skip" and d.residue in by
            and not (d.action == "enhanced_sampling" and handed)]
    if not live:
        return 0.0
    precision = state.precision()
    items = [ItemState(d.residue, by[d.residue].standard_error or {}, by[d.residue].error_within or 0.0,
                       by[d.residue].error_between or 0.0, by[d.residue].transitions, by[d.residue].verdict == "ok",
                       by[d.residue].burn_in_ns or 0.0)
             for d in live]
    alloc = state.notes.setdefault("allocation", {"looks": 0, "tests": 0})
    if alloc.get("counting") != 2:
        # Runs started before 2026-11-20 counted every look after the first, also looks that could not stop
        # because the state changes were too few. Count them again from the recorded looks.
        alloc["tests"] = recount_tests(state.notes.get("actions", []))
        alloc["counting"] = 2
    limits = Limits(max_replicas=goal.max_replicas if state.allowed("add_replicas") else n,
                    max_ns=state.max_ns_per_replica(), budget_ns=state.budget_ns(), chunk_ns=state.loop.extend_ns,
                    patience_ns=state.loop.no_flip_patience_ns or 0.0,   # None (default): the budget rule only.
                    stop_looks=goal.stop_looks,
                    min_transitions=state.loop.min_flips_per_replica,
                    t_dist=precision.criterion == "confidence")
    look = alloc["looks"] + 1
    step = next_step(items, n, ns, look, alloc["tests"], precision.z, goal.precision, limits,
                     state.md.equilibration_ps / 1000.0)
    alloc["looks"] = look
    alloc["tests"] += 1 if step.counted else 0
    what = f"{n} × {ns:g} ns"
    why_zh = STEP_ZH.get(step.kind, step.reason)
    stuck = [k for k, v in step.set_aside.items() if v == "stuck"]
    if step.kind == "stuck" and stuck:
        from .allocation import stuck_projection

        rate, need, data = stuck_projection(n, ns, precision.z, goal.precision)
        rest = max(limits.budget_ns - n * ns, 0.0)
        why_zh = (f"{', '.join(stuck)} 在所有重复加起来 {n * ns:g} ns 里一次都没有变化。按这个结果，变化最快也只能约每 "
                  f"{1 / rate:.3g} ns 一次（95% 置信上限）；要达到你要的精度，大约需要 {need:.0f} 次变化，也就是还要至少 "
                  f"{data:.3g} ns，超过了预算 {limits.budget_ns:g} ns 里剩下的 {rest:.3g} ns。再跑普通 MD 也得不到可靠的比例")
    # Items plain MD cannot answer are set aside on their own: told once, marked for the report, while the
    # other items go on to their precision.
    for d in live:
        if d.residue in step.set_aside:
            d.action, d.cause = "stalled", ("stalled" if step.set_aside[d.residue] == "stuck" else "insufficient_evidence")
    others_go_on = any(d.residue not in step.set_aside for d in live)
    for label in [k for k in list(state.notes) if k.startswith("said_set_aside_")]:
        if label[len("said_set_aside_"):] not in step.set_aside:
            state.notes.pop(label)   # It moved again: tell anew if it is ever set aside again.
    for label, why in step.set_aside.items():
        if state.notes.get(f"said_set_aside_{label}") or not others_go_on:
            continue   # Told once is enough; and when nothing goes on, the stop message says it all.
        stuck = why == "stuck"
        record(run_dir, state, Action("plain_md", "set_aside", True,
                                      f"{label}: INSUFFICIENT_EVIDENCE on its own ("
                                      + (f"never changed state in {ns:g} ns per replica" if stuck else
                                         "its projected cost is far beyond the budget")
                                      + "); the other items go on.", ns),
               f"{name}：{label} "
               + (f"每条跑了 {ns:g} ns 一次都没有变化" if stuck else "按投影要远超预算才能达到精度")
               + "，单独标为证据不足（INSUFFICIENT_EVIDENCE）。其余项目照常继续，不受它影响。",
               f"{name}: {label} "
               + (f"never changed state in {ns:g} ns per replica" if stuck else
                  "would need far more than the budget to reach the precision")
               + ", so it gets INSUFFICIENT_EVIDENCE on its own. The other items go on as planned.",
               "suggestion", once_key=f"said_set_aside_{label}")
    if step.action == STOP:
        room = round(state.loop.max_ns - ns, 6)
        if not state.allowed("stop_early") and room > 0:
            record(run_dir, state, Action("plain_md", "extend", True, "Precise enough, but stopping early is "
                                          "not allowed; running to the planned length.", ns))
            return min(state.loop.extend_ns, room)
        state.notes["allocation_decision"] = {"decision": STOP, "kind": step.kind, "reason": step.reason,
                                              "look": look, "replicas": n, "ns_per_replica": ns}
        record(run_dir, state, Action("plain_md", "stop", True, f"STOP at look {look} ({what}): {step.reason}.", ns),
               f"{name}：第 {look} 次查看（{what}），决定：停止（STOP）。{why_zh}。",
               f"{name}: look {look} ({what}). Decision: STOP. {step.reason[0].upper() + step.reason[1:]}.")
        return 0.0
    if step.action == INSUFFICIENT:
        for d in live:
            if by[d.residue].verdict == "ok" or d.residue in step.set_aside:
                continue
            stuck = sum(by[d.residue].transitions) == 0
            to_es = state.can_switch() and state.enhanced.get("enabled", True)
            d.action = "enhanced_sampling" if to_es else "stalled"
            d.cause = "stalled" if stuck else "insufficient_evidence"
        state.notes["allocation_decision"] = {"decision": INSUFFICIENT, "kind": step.kind, "reason": step.reason,
                                              "look": look, "replicas": n, "ns_per_replica": ns}
        record(run_dir, state, Action("plain_md", "insufficient_evidence", True,
                                      f"INSUFFICIENT_EVIDENCE at look {look} ({what}): {step.reason}.", ns),
               f"{name}：第 {look} 次查看（{what}），决定：证据不足（INSUFFICIENT_EVIDENCE）。{why_zh}。"
               "这是一个结论，不是失败：现有数据撑不起要求的精度。",
               f"{name}: look {look} ({what}). Decision: INSUFFICIENT_EVIDENCE. "
               f"{step.reason[0].upper() + step.reason[1:]}. That is an answer, not a failure: the data cannot "
               "support the asked precision.", "suggestion")
        return 0.0
    assert step.action == CONTINUE
    if state.notes.pop("allocation_decision", None) is not None:
        state.notes.pop("plain_done", None)   # A run that had ended goes on: tell its new end again.
    added = step.replicas - n
    if added > 0:
        state.n_replicas = step.replicas
    plan = f"{step.replicas} × {step.ns:g} ns"
    record(run_dir, state, Action("plain_md", "add_replicas" if added > 0 else "extend", True,
                                  f"Look {look}{' (planning only)' if step.planning else ''} at {what}: {step.reason}. "
                                  f"Next: {plan}.", ns),
           f"{name}：第 {look} 次查看（{what}{'，只做规划，不会在这次停' if step.planning else ''}）。"
           f"决定：{'增加 ' + str(added) + ' 条重复（先补跑到同样长度），' if added > 0 else ''}"
           f"每条跑到 {step.ns:g} ns，共 {plan}。理由：{why_zh}。",
           f"{name}: look {look} at {what}{' (planning only: this look may not stop)' if step.planning else ''}. "
           f"Decision: {'add ' + str(added) + ' replicas (they first catch up), ' if added > 0 else ''}"
           f"run every replica to {step.ns:g} ns, {plan} in total. Reason: {step.reason}.")
    return round(step.ns - ns, 6)
