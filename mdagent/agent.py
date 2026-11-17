"""The adaptive loop: the agent decides how long to run.

The agent runs MD in rounds. After each round it diagnoses the target side
chains and makes one decision for each residue:

- "done": the side chain flipped often enough. No more work.
- "extend": plain MD is still useful (it saw flips, or it has not run long
  enough to give up). Run all replicas longer.
- "enhanced_sampling": plain MD did not see a flip after a long time, or the
  time budget is used up. Plain MD will not answer the question.
- "skip": the residue has no chi1 (Gly or Ala).

If any residue is "extend" and the budget allows, the agent runs one more
round for all replicas. Otherwise it stops.

Two modes:
- "auto" (default): the agent chooses. When plain MD makes no progress, it
  switches to enhanced sampling by itself, and it stops when more sampling no
  longer changes the result.
- "guided": the user chose the method. The agent runs only that method, within
  the user's budget. When it makes no progress, the agent posts a suggestion in
  the chat (messages.py) and goes on; it never switches by itself.

All state is in the run directory, so a stopped run can resume. A round
that was stopped halfway is finished first; completed parts are not rerun.
Files:
- run.json      settings, target, and the length of each round
- status.json   the latest diagnosis and decisions (for other programs to read)
- decisions.log the decisions with reasons, in plain text
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .diagnose import ResidueReport, diagnose_trajectories, fixed_chi1_residues, format_report, load_run
from .adaptive import AdaptiveSettings, choose_starts
from .messages import post
from .simulate import MDSettings, SystemFiles, load_inputs, part_path, run_segment


@dataclass
class LoopSettings:
    first_ns: float = 1.0           # Length of round 1, per replica.
    extend_ns: float = 5.0          # Length of each later round, per replica.
    max_ns: float = 20.0            # Budget: total length per replica.
    no_flip_patience_ns: float = 10.0  # Give up on plain MD for a residue with no flip after this.
    min_flips_per_replica: int = 4


@dataclass
class RunState:
    pdb: str
    target_description: str
    target: list[tuple[str, int]]
    n_replicas: int
    base_seed: int
    md: MDSettings
    loop: LoopSettings
    rounds_ns: list[float] = field(default_factory=list)
    pending_round_ns: float = 0.0  # Length of a round that started but did not finish.
    setup: dict = field(default_factory=dict)  # The SetupOptions used to build the system.
    # Enhanced sampling: {"enabled": bool, "settings": MetadSettings fields,
    # "residues": [[chain, number], ...], "resnames": {"A:111": "VAL"}}
    enhanced: dict = field(default_factory=lambda: {"enabled": True})
    # Plain MD mode: "continue" (each replica continues its own trajectory) or
    # "adaptive" (from round 2, AdaptivePy chooses the starting frames).
    sampling: str = "continue"
    mode: str = "auto"      # "auto": the agent chooses methods and when to stop. "guided": the user chose.
    notes: dict = field(default_factory=dict)     # Which messages were posted already, so each is posted once.
    adaptive: dict = field(default_factory=dict)  # AdaptiveSettings fields
    starts: dict = field(default_factory=dict)    # Round index -> [[replica, part, frame], ...]

    @property
    def ns_done(self) -> float:
        return sum(self.rounds_ns)

    def save(self, run_dir: Path) -> None:
        _write_json(run_dir / "run.json", asdict(self))

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        d = json.loads((run_dir / "run.json").read_text())
        d["md"] = MDSettings(**d["md"])
        d["loop"] = LoopSettings(**d["loop"])
        d["target"] = [tuple(k) for k in d["target"]]
        return cls(**d)


@dataclass
class Decision:
    residue: str
    action: str
    reason: str
    key: tuple = ()  # (chain, residue number)
    stalled: bool = False  # Guided mode: plain MD makes no progress, but goes on because the user chose it.


def decide(reports: list[ResidueReport], ns_done: float, loop: LoopSettings,
           mode: str = "auto") -> tuple[list[Decision], float]:
    """Return a decision for each residue, and the length of the next round (0 means stop).

    In guided mode a residue with no flip after the patience time does not stop plain MD:
    the user chose plain MD, so it goes on until the budget (the decision is marked stalled).
    """
    room = round(loop.max_ns - ns_done, 6)
    decisions = []
    for r in reports:
        if r.verdict in ("no_chi1", "fixed_chi1"):
            decisions.append(Decision(r.label, "skip", r.reason))
        elif r.verdict == "ok":
            decisions.append(Decision(r.label, "done", f"Sampled well: {r.reason}"))
        elif sum(r.transitions) == 0 and ns_done >= loop.no_flip_patience_ns and mode == "guided" and room > 0:
            decisions.append(Decision(
                r.label, "extend",
                f"No flip in {ns_done:g} ns per replica. Plain MD is unlikely to see one, but the user chose "
                "plain MD, so it goes on until the budget.", stalled=True))
        elif sum(r.transitions) == 0 and ns_done >= loop.no_flip_patience_ns:
            decisions.append(Decision(
                r.label, "enhanced_sampling",
                f"No flip in {ns_done:g} ns per replica. Plain MD is unlikely to see one."))
        elif room <= 0:
            decisions.append(Decision(
                r.label, "enhanced_sampling",
                f"The budget of {loop.max_ns:g} ns per replica is used up. {r.reason}"))
        elif sum(r.transitions) == 0:
            decisions.append(Decision(
                r.label, "extend",
                f"No flip yet, but only {ns_done:g} ns per replica so far. Run longer first. {r.reason}"))
        else:
            decisions.append(Decision(
                r.label, "extend", f"Plain MD sees flips, but not enough yet. {r.reason}"))

    for d, r in zip(decisions, reports):
        d.key = (r.chain, r.resseq)
    if any(d.action == "extend" for d in decisions):
        return decisions, min(loop.extend_ns, room)
    return decisions, 0.0


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _record(run_dir: Path, state: RunState, reports, decisions, next_ns: float) -> str:
    lines = [f"Round {len(state.rounds_ns)} finished. "
             f"Total: {state.ns_done:g} ns per replica, {state.n_replicas} replicas."]
    for d in decisions:
        if d.action in ("extend", "enhanced_sampling"):
            lines.append(f"  {d.residue}: {d.action}. {d.reason}")
    n_done = sum(d.action == "done" for d in decisions)
    if n_done:
        lines.append(f"  {n_done} residue(s) are sampled well.")
    if next_ns > 0:
        lines.append(f"Decision: run every replica {next_ns:g} ns longer.")
    else:
        enhanced = [d.residue for d in decisions if d.action == "enhanced_sampling"]
        lines.append("Decision: stop plain MD. "
                     + (f"Enhanced sampling is needed for: {', '.join(enhanced)}."
                        if enhanced else "All target side chains are sampled well."))
    text = "\n".join(lines)
    with open(run_dir / "decisions.log", "a") as f:
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + text + "\n\n")

    _write_json(run_dir / "status.json", {
        "round": len(state.rounds_ns),
        "ns_per_replica": state.ns_done,
        "next_round_ns": next_ns,
        "finished": next_ns == 0,
        "residues": [
            {**asdict(r), "rank": i + 1, "time_between_flips_ns": r.time_between_flips_ns,
             "decision": d.action, "decision_reason": d.reason}
            for i, (r, d) in enumerate(zip(reports, decisions))
        ],
    })
    return text


def run_loop(
    run_dir: Path,
    state: RunState,
    files: SystemFiles | None = None,
    log: Callable[[str], None] = print,
) -> list[Decision]:
    """Run rounds until the agent decides to stop. Resumes from what is on disk."""
    files = files or load_inputs(run_dir)
    if state.pending_round_ns > 0:
        next_ns = state.pending_round_ns  # Finish the round that was stopped.
    elif state.rounds_ns:
        reports, decisions, next_ns = _diagnose_and_decide(run_dir, state)
        log(_record(run_dir, state, reports, decisions, next_ns))
    else:
        next_ns = state.loop.first_ns

    while next_ns > 0:
        part = len(state.rounds_ns)
        state.pending_round_ns = next_ns
        state.save(run_dir)
        log(f"\nRound {part + 1}: running {state.n_replicas} replicas for {next_ns:g} ns each.")
        adaptive = state.sampling == "adaptive"
        full_ps = AdaptiveSettings.from_dict(state.adaptive).full_interval_ps if adaptive else None
        starts = _adaptive_starts(run_dir, state, part, log) if adaptive and part > 0 else None
        for i in range(state.n_replicas):
            if part_path(run_dir, i, part, ".state.xml").exists():
                log(f"replica {i}: round {part + 1} is already complete.")
                continue
            seed = state.base_seed + 1000 * part + i
            start = None
            if starts:
                r, p, frame = starts[i]
                start = (part_path(run_dir, r, p, ".full.dcd"), frame)
            run_segment(files, run_dir, i, part, seed, next_ns, state.md, log,
                        start_frame=start, full_interval_ps=full_ps)
        state.rounds_ns.append(next_ns)
        state.pending_round_ns = 0.0
        state.save(run_dir)

        reports, decisions, next_ns = _diagnose_and_decide(run_dir, state)
        log("\n" + format_report(reports) + "\n")
        log(_record(run_dir, state, reports, decisions, next_ns))
    return decisions


def _adaptive_starts(run_dir: Path, state: RunState, part: int, log) -> list[list[int]]:
    """Starting frames for round `part`, chosen once by AdaptivePy and kept in run.json."""
    key = str(part)
    if key not in state.starts:
        segments = [(r, p) for r in range(state.n_replicas) for p in range(part)]
        chosen = choose_starts(run_dir, part, segments, state.target, state.n_replicas,
                               AdaptiveSettings.from_dict(state.adaptive),
                               random_seed=state.base_seed + part, log=log)
        state.starts[key] = [list(c) for c in chosen]
        state.save(run_dir)
    return state.starts[key]


def _diagnose_and_decide(run_dir: Path, state: RunState):
    fixed = fixed_chi1_residues(run_dir / "protein.pdb")
    if state.sampling == "adaptive":
        # Segments start from different frames on purpose, so judge the pooled flips.
        reports = diagnose_trajectories(
            load_run(run_dir, state.target, len(state.rounds_ns), join=False), state.target,
            state.ns_done, min_flips_per_replica=state.loop.min_flips_per_replica,
            pooled_total_ns=state.ns_done * state.n_replicas, n_replicas=state.n_replicas, fixed=fixed)
    else:
        reports = diagnose_trajectories(
            load_run(run_dir, state.target, len(state.rounds_ns)), state.target, state.ns_done,
            min_flips_per_replica=state.loop.min_flips_per_replica, fixed=fixed)
    decisions, next_ns = decide(reports, state.ns_done, state.loop, state.mode)
    _post_plain_md_messages(run_dir, state, decisions, next_ns)
    return reports, decisions, next_ns


def _once(run_dir: Path, state: RunState, key: str, zh: str, en: str, kind: str) -> None:
    """Post a message to the chat, but only once per run."""
    if state.notes.get(key):
        return
    post(run_dir, zh, en, kind)
    state.notes[key] = True
    state.save(run_dir)


def _post_plain_md_messages(run_dir: Path, state: RunState, decisions: list[Decision], next_ns: float) -> None:
    stalled = [d.residue for d in decisions if d.stalled]
    if stalled:
        names = ", ".join(stalled)
        _once(run_dir, state, "stalled",
              f"普通 MD 已经跑了每条 {state.ns_done:g} ns，{names} 的侧链一次都没有翻转。继续跑普通 MD 很可能也看不到翻转。"
              f"建议改用增强采样。运行会按你的设置继续，直到每条 {state.loop.max_ns:g} ns。"
              f"要换方法，可以说“停止 {run_dir.name}”，然后说“用增强采样继续 {run_dir.name}”。",
              f"Plain MD has run {state.ns_done:g} ns per replica, and the side chains of {names} have not flipped "
              f"once. More plain MD is unlikely to show a flip. I suggest enhanced sampling. The run goes on with your "
              f"settings until {state.loop.max_ns:g} ns per replica. To switch, say \"stop {run_dir.name}\", then "
              f"\"continue {run_dir.name} with enhanced sampling\".", "suggestion")
    if next_ns > 0:
        return
    need = [d.residue for d in decisions if d.action == "enhanced_sampling"]
    if not need:
        _once(run_dir, state, "plain_done",
              f"普通 MD 结束（每条 {state.ns_done:g} ns）：目标区域的侧链都采样充分了，再跑不会有明显增益。",
              f"Plain MD finished ({state.ns_done:g} ns per replica): all target side chains are sampled well, so "
              "more sampling would add little.", "done")
    elif state.mode == "guided" and not state.enhanced.get("requested"):
        _once(run_dir, state, "plain_done",
              f"普通 MD 按你的设置结束（每条 {state.ns_done:g} ns）。这些侧链没有采样充分：{', '.join(need)}。"
              f"建议用增强采样继续，可以说“用增强采样继续 {run_dir.name}”。",
              f"Plain MD finished as you set it ({state.ns_done:g} ns per replica). These side chains are not "
              f"sampled well: {', '.join(need)}. I suggest continuing with enhanced sampling: say \"continue "
              f"{run_dir.name} with enhanced sampling\".", "suggestion")
    elif state.enhanced.get("enabled", True):
        _once(run_dir, state, "plain_done",
              f"普通 MD 结束（每条 {state.ns_done:g} ns）。这些侧链普通 MD 采样不到：{', '.join(need)}。"
              "现在自动改用增强采样（chi1 上的 metadynamics）。",
              f"Plain MD finished ({state.ns_done:g} ns per replica). Plain MD cannot sample these side chains: "
              f"{', '.join(need)}. Switching to enhanced sampling (metadynamics on chi1) now.", "info")


def run_all(
    run_dir: Path,
    state: RunState,
    files: SystemFiles | None = None,
    log: Callable[[str], None] = print,
) -> tuple[list[Decision], list[str]]:
    """Plain MD until the agent stops it, then enhanced sampling where it is needed.

    Returns the plain MD decisions and the labels of the residues that got enhanced sampling.
    """
    from .enhanced import MetadSettings, ns_done, run_replica
    from .reweight import StopRule, analyze, converged, format_table
    from .reweight import save as save_populations

    files = files or load_inputs(run_dir)
    decisions = run_loop(run_dir, state, files, log)
    wanted = [d for d in decisions if d.action == "enhanced_sampling"]
    if not wanted:
        return decisions, []
    if state.mode == "guided" and not state.enhanced.get("requested"):
        log("Guided mode: the user chose plain MD, so enhanced sampling does not start by itself. "
            "Use 'resume --enhanced-sampling' to run it.")
        return decisions, []
    if not state.enhanced.get("enabled", True):
        log("Enhanced sampling is switched off for this run. Use 'resume --enhanced-sampling' to run it.")
        return decisions, []

    es = state.enhanced
    if es.get("residues"):
        residues = [tuple(k) for k in es["residues"]]  # Keep the set that was started before.
        if sorted(residues) != sorted(d.key for d in wanted):
            log("Note: enhanced sampling already started for "
                + ", ".join(f"{c}:{n}" for c, n in residues) + ". It continues with these residues.")
    else:
        settings = MetadSettings.from_dict(es.get("settings", {}))
        # Decisions follow the diagnosis ranking, so the slowest side chains come first.
        chosen, left_out = wanted[:settings.max_residues], wanted[settings.max_residues:]
        if left_out:
            log(f"Note: {len(wanted)} residues need enhanced sampling, but at most {settings.max_residues} are "
                f"biased at once. Left out (faster ones): {', '.join(d.residue for d in left_out)}. "
                "To bias more, start a new run with a larger --enhanced-max-residues, or a smaller target region.")
        residues = [d.key for d in chosen]
        es["residues"] = [list(k) for k in residues]
        es["resnames"] = {f"{d.key[0]}:{d.key[1]}": d.residue.rstrip("0123456789") for d in chosen}
    settings = MetadSettings.from_dict(es.get("settings", {}))
    es["settings"] = asdict(settings)
    n_rep = min(settings.n_replicas, state.n_replicas)
    state.save(run_dir)

    labels = [f"{es['resnames'][f'{c}:{n}']}{n}" for c, n in residues]
    rule = StopRule(**{**asdict(StopRule()), **es.get("stop_rule", {})})
    rule.min_ns = min(rule.min_ns, settings.length_ns)  # A short test run may stop at its own length.
    auto = state.mode != "guided"
    log(f"\nEnhanced sampling (well-tempered metadynamics on chi1) for {', '.join(labels)}: "
        f"{n_rep} replicas, up to {settings.length_ns:g} ns each, starting from the end of plain MD. "
        + ("It stops earlier when more sampling no longer changes the populations." if auto else
           "Guided mode: it runs the full length and says when the populations stop changing."))
    last = len(state.rounds_ns) - 1
    done = min(ns_done(run_dir, r) for r in range(n_rep))
    result = before = None
    stopped_early = False
    finished_before = bool(es.get("converged")) and auto
    if finished_before:  # Resumed after it had already stopped: report the saved result again.
        result = analyze(run_dir, labels, n_rep, settings, state.md.temperature_k, rule)
        stopped_early = done < settings.length_ns - 1e-9
    while not finished_before:
        if done < settings.length_ns - 1e-9:
            target = min(done + settings.chunk_ns, settings.length_ns)
            for r in range(n_rep):  # The replicas take turns, so they can be compared after each round.
                start = part_path(run_dir, r, last, ".state.xml")
                run_replica(files, run_dir, r, start, residues, settings, state.md,
                            seed=state.base_seed + 50000 + r, log=log, until_ns=target)
            done = target
        before, result = result, analyze(run_dir, labels, n_rep, settings, state.md.temperature_k, rule)
        if result is None:
            break
        save_populations(run_dir, result)
        ok, still = converged(result, before, rule)
        log("\n" + format_table(result) + "\n" + (
            "Populations no longer change: more sampling would add little." if ok else
            f"Still changing or replicas disagree: {', '.join(still) or '(too short to judge)'}."))
        if ok:
            es["converged"] = True
            state.save(run_dir)
            if auto:
                stopped_early = done < settings.length_ns - 1e-9
                break
            _once(run_dir, state, "enhanced_converged",
                  f"增强采样的结果已经稳定（每条 {done:g} ns）：再跑各状态的比例也不会明显变化。"
                  f"运行会按你的设置跑满 {settings.length_ns:g} ns；想提前结束可以说“停止 {run_dir.name}”。",
                  f"The enhanced sampling result is stable ({done:g} ns per replica): more sampling would not "
                  f"change the populations much. The run goes on to {settings.length_ns:g} ns as you set it; say "
                  f"\"stop {run_dir.name}\" to end it now.", "info")
        if done >= settings.length_ns - 1e-9:
            break

    folder = (run_dir / "enhanced").resolve()
    if result is not None and es.get("converged"):
        why = (f"stopped at {done:g} ns per replica because more sampling no longer changed the populations"
               if stopped_early else f"ran {done:g} ns per replica; the populations are stable")
        post(run_dir,
             f"结果已经可以用了：增强采样{'提前停下' if stopped_early else '跑完了'}（每条 {done:g} ns），"
             f"各条轨迹的结果一致、继续跑也不再变化。重新加权后的各状态比例在 {folder}/populations.json，"
             "日志末尾也有一张表。",
             f"Results are ready: enhanced sampling {why}, and the replicas agree. The reweighted state "
             f"populations are in {folder}/populations.json and in a table at the end of the log.", "done")
    elif result is not None:
        _, still = converged(result, before, rule)
        why = f"ran the full {done:g} ns per replica; still changing: {', '.join(still)}"
        post(run_dir,
             f"增强采样跑满了每条 {done:g} ns，但这些侧链的结果还在变化，或各条轨迹不一致：{', '.join(still)}。"
             f"现有结果在 {folder}/populations.json，误差可能较大。可以加长增强采样，"
             f"例如说“把 {run_dir.name} 的增强采样加长到 {2 * settings.length_ns:g} ns”。"
             "如果一直不收敛，这些侧链可能和别的慢运动耦合在一起，只偏置 chi1 推不动。",
             f"Enhanced sampling ran the full {done:g} ns per replica, but these side chains still change or the "
             f"replicas disagree: {', '.join(still)}. The current result is in {folder}/populations.json and may "
             f"have large errors. You can run longer, for example \"run the enhanced sampling of {run_dir.name} to "
             f"{2 * settings.length_ns:g} ns\". If it never settles, these side chains may be coupled to another "
             "slow motion that a chi1 bias does not push.", "suggestion")
    else:
        why = f"ran {done:g} ns per replica (too short to reweight)"
    text = f"Enhanced sampling for {', '.join(labels)}: {why}. Output in {folder}."
    log(text)
    with open(run_dir / "decisions.log", "a") as f:
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + text + "\n\n")
    return decisions, labels
