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
    goal: dict = field(default_factory=dict)      # Goal fields. Empty in runs made before goals existed.
    distances: list = field(default_factory=list)  # Distance specs (observables.py) for quantity distance_states.
    adaptive: dict = field(default_factory=dict)  # AdaptiveSettings fields
    starts: dict = field(default_factory=dict)    # Round index -> [[replica, part, frame], ...]

    @property
    def ns_done(self) -> float:
        return sum(self.rounds_ns)

    def the_goal(self) -> Goal:
        if not self.goal:
            return Goal.for_mode(self.mode)  # Older runs: the defaults of their mode.
        names = set(Goal.__dataclass_fields__)
        return Goal(**{k: v for k, v in self.goal.items() if k in names})

    @property
    def quantity(self) -> str:
        return self.the_goal().quantity

    def can_switch(self) -> bool:
        """May the agent switch to enhanced sampling by itself? Needs the permission, and a quantity that
        enhanced sampling can push (only chi1 for now)."""
        return self.allowed("switch_method") and self.quantity == "chi1_populations"

    def allowed(self, action: str) -> bool:
        if action == "switch_method" and self.enhanced.get("requested"):
            return True  # The user asked for enhanced sampling (resume --enhanced-sampling).
        return action in self.the_goal().permissions

    def save(self, run_dir: Path) -> None:
        _write_json(run_dir / "run.json", asdict(self))

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        d = json.loads((run_dir / "run.json").read_text())
        d["md"] = MDSettings(**d["md"])
        d["loop"] = LoopSettings(**d["loop"])
        d["target"] = [tuple(k) for k in d["target"]]
        return cls(**d)


PERMISSIONS = ("extend", "add_replicas", "switch_method", "stop_early")
DEFAULT_PERMISSIONS = {
    # auto: the agent may do everything by itself.
    "auto": list(PERMISSIONS),
    # guided: the user chose the method and the budget; the agent only suggests.
    "guided": [],
}


@dataclass
class Goal:
    """What the user wants to know, how precisely, and what the agent may do by itself.

    quantity: the result the user cares about. Supported now: "chi1_populations" (the state
      populations of the target side chains).
    precision: the largest allowed standard error of any state population (0 to 1).
    permissions: actions the agent may take without asking:
      extend         run longer than the planned budget when the goal is not met yet
      add_replicas   start more independent replicas
      switch_method  switch from plain MD to enhanced sampling
      stop_early     stop before the budget once the goal is met
    """

    quantity: str = "chi1_populations"   # or "distance_states" (observables.py)
    description: str = ""          # The user's own words.
    precision: float = 0.05
    permissions: list = field(default_factory=lambda: list(PERMISSIONS))
    max_replicas: int = 10          # add_replicas stops here (literature: 5 to 10 replicas).
    extend_factor: float = 2.0      # extend may run to this many times the first budget.

    @classmethod
    def for_mode(cls, mode: str, **kw) -> "Goal":
        return cls(permissions=list(DEFAULT_PERMISSIONS[mode]), **kw)


@dataclass
class Decision:
    residue: str
    action: str
    reason: str
    key: tuple = ()  # (chain, residue number)
    stalled: bool = False  # Guided mode: plain MD makes no progress, but goes on because the user chose it.
    cause: str = ""        # For "enhanced_sampling": "stalled" (no flip after the patience time) or
                           # "budget_exhausted" (the plain MD budget is used up first).


def decide(reports: list[ResidueReport], ns_done: float, loop: LoopSettings,
           may_switch: bool = True, bias_available: bool = True) -> tuple[list[Decision], float]:
    """Return a decision for each item (residue or distance), and the length of the next round (0: stop).

    Without permission to switch methods, an item with no state change after the patience time does
    not stop plain MD: the user chose plain MD, so it goes on until the budget (marked stalled).
    Without an enhanced sampling method for the quantity, such an item gets the action "stalled"
    instead of "enhanced_sampling" when plain MD stops.
    """
    need = "enhanced_sampling" if bias_available else "stalled"
    room = round(loop.max_ns - ns_done, 6)
    decisions = []
    for r in reports:
        if r.verdict in ("no_chi1", "fixed_chi1"):
            decisions.append(Decision(r.label, "skip", r.reason))
        elif r.verdict == "ok":
            decisions.append(Decision(r.label, "done", f"Sampled well: {r.reason}"))
        elif sum(r.transitions) == 0 and ns_done >= loop.no_flip_patience_ns and not (may_switch and bias_available) \
                and room > 0:
            decisions.append(Decision(
                r.label, "extend",
                f"No flip in {ns_done:g} ns per replica. Plain MD is unlikely to see one, but the user chose "
                "plain MD, so it goes on until the budget.", stalled=True))
        elif sum(r.transitions) == 0 and ns_done >= loop.no_flip_patience_ns:
            decisions.append(Decision(
                r.label, need,
                f"No flip in {ns_done:g} ns per replica. Plain MD is unlikely to see one.", cause="stalled"))
        elif room <= 0:
            decisions.append(Decision(
                r.label, need,
                f"The budget of {loop.max_ns:g} ns per replica is used up. {r.reason}", cause="budget_exhausted"))
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
        if d.action in ("extend", "enhanced_sampling", "stalled"):
            lines.append(f"  {d.residue}: {d.action}. {d.reason}")
    n_done = sum(d.action == "done" for d in decisions)
    if n_done:
        lines.append(f"  {n_done} item(s) are sampled well.")
    if next_ns > 0:
        lines.append(f"Decision: run every replica {next_ns:g} ns longer.")
    else:
        enhanced = [d.residue for d in decisions if d.action == "enhanced_sampling"]
        stuck = [d.residue for d in decisions if d.action == "stalled"]
        lines.append("Decision: stop plain MD. "
                     + (f"Enhanced sampling is needed for: {', '.join(enhanced)}. " if enhanced else "")
                     + (f"Not sampled well, and this quantity has no enhanced sampling method yet: "
                        f"{', '.join(stuck)}." if stuck else "")
                     + ("All target items are sampled well." if not enhanced and not stuck else ""))
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
            if not adaptive:
                for p in range(part):  # A replica added later (policy: add_replicas) first catches up.
                    if not part_path(run_dir, i, p, ".state.xml").exists():
                        log(f"replica {i}: catching up with round {p + 1} ({state.rounds_ns[p]:g} ns).")
                        run_segment(files, run_dir, i, p, state.base_seed + 1000 * p + i, state.rounds_ns[p],
                                    state.md, log)
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


def diagnose_run(run_dir: Path, state: RunState) -> list[ResidueReport]:
    """Diagnose the plain MD of a run for its quantity (chi1 or distances). The run and `mdagent diagnose`
    both use this, so they always judge the same way."""
    if state.quantity == "distance_states":
        return _diagnose_distances(run_dir, state)
    fixed = fixed_chi1_residues(run_dir / "protein.pdb")
    if state.sampling == "adaptive":
        # Segments start from different frames on purpose, so judge the pooled flips.
        return diagnose_trajectories(
            load_run(run_dir, state.target, len(state.rounds_ns), join=False), state.target,
            state.ns_done, min_flips_per_replica=state.loop.min_flips_per_replica,
            pooled_total_ns=state.ns_done * state.n_replicas, n_replicas=state.n_replicas, fixed=fixed)
    return diagnose_trajectories(
        load_run(run_dir, state.target, len(state.rounds_ns)), state.target, state.ns_done,
        min_flips_per_replica=state.loop.min_flips_per_replica, fixed=fixed,
        precision=state.the_goal().precision)


def _diagnose_and_decide(run_dir: Path, state: RunState):
    reports = diagnose_run(run_dir, state)
    decisions, next_ns = decide(reports, state.ns_done, state.loop, state.allowed("switch_method"),
                                bias_available=state.quantity != "distance_states")
    next_ns = _apply_policy(run_dir, state, reports, decisions, next_ns)
    _post_plain_md_messages(run_dir, state, decisions, next_ns)
    return reports, decisions, next_ns


def _apply_policy(run_dir: Path, state: RunState, reports, decisions, next_ns: float) -> float:
    """Choose the next action (policy.py) once per round; a resumed run keeps the choice it made."""
    from .policy import plain_md_next

    key = {"ns": state.ns_done, "max_ns": state.loop.max_ns, "n_replicas": state.n_replicas,
           "permissions": sorted(state.the_goal().permissions)}
    done = state.notes.get("policy_round")
    if done and done.get("key") == key:
        return done["next_ns"]  # Same round, same budget: keep the choice (no second message, no second replica).
    next_ns = plain_md_next(run_dir, state, reports, decisions, next_ns)
    key.update(max_ns=state.loop.max_ns, n_replicas=state.n_replicas)  # As the policy left them.
    state.notes["policy_round"] = {"key": key, "next_ns": next_ns}
    state.save(run_dir)
    return next_ns


def _diagnose_distances(run_dir: Path, state: RunState) -> list[ResidueReport]:
    """Distances: load only their atoms (sorted, as mdtraj keeps them) and judge each one."""
    import mdtraj as md

    from .observables import diagnose_distances, distance_atoms, parse_distances

    specs = parse_distances(state.distances)
    pairs = distance_atoms(md.load_topology(str(run_dir / "protein.pdb")), specs)
    atoms = sorted({i for pair in pairs for i in pair})
    where = {a: k for k, a in enumerate(atoms)}
    trajs = load_run(run_dir, state.target, len(state.rounds_ns), atoms=atoms)
    return diagnose_distances(trajs, specs, [(where[i], where[j]) for i, j in pairs], state.ns_done,
                              state.loop.min_flips_per_replica, state.the_goal().precision)


def _once(run_dir: Path, state: RunState, key: str, zh: str, en: str, kind: str) -> None:
    """Post a message to the chat, but only once per run."""
    if state.notes.get(key):
        return
    post(run_dir, zh, en, kind)
    state.notes[key] = True
    state.save(run_dir)


def _advice(state: RunState, run_dir: Path) -> tuple[str, str]:
    """What to try next when plain MD does not sample an item, for this quantity (Chinese, English)."""
    if state.quantity == "chi1_populations":
        return (f"建议改用增强采样，可以说“用增强采样继续 {run_dir.name}”（还在运行时先说“停止 {run_dir.name}”）。",
                f"I suggest enhanced sampling: say \"continue {run_dir.name} with enhanced sampling\" (if it is still "
                f"running, first say \"stop {run_dir.name}\").")
    return ("这个量还没有增强采样方法。可以加长普通 MD、增加重复，或者新开一个运行改用自适应采样（--sampling adaptive）；"
            "如果这个距离本来就很少变化，也可能它只有一个稳定状态。",
            "This quantity has no enhanced sampling method yet. You can run plain MD longer, add replicas, or start a "
            "new run with adaptive sampling (--sampling adaptive). The distance may also have only one stable state.")


def _post_plain_md_messages(run_dir: Path, state: RunState, decisions: list[Decision], next_ns: float) -> None:
    stalled = [d.residue for d in decisions if d.stalled]
    advice_zh, advice_en = _advice(state, run_dir)
    if stalled:
        names = ", ".join(stalled)
        _once(run_dir, state, "stalled",
              f"普通 MD 已经跑了每条 {state.ns_done:g} ns，{names} 一次都没有在状态之间变化。继续跑普通 MD 很可能也看不到。"
              f"运行会按你的设置继续，直到每条 {state.loop.max_ns:g} ns。" + advice_zh,
              f"Plain MD has run {state.ns_done:g} ns per replica, and {names} have not changed state once. More plain "
              f"MD is unlikely to show it. The run goes on with your settings until {state.loop.max_ns:g} ns per "
              "replica. " + advice_en, "suggestion")
    if next_ns > 0:
        return
    need = [d.residue for d in decisions if d.action in ("enhanced_sampling", "stalled")]
    # Why plain MD stopped. Stalled wins over budget: it says that more plain MD would not help.
    reason = ("target_reached" if not need else
              "stalled" if any(d.cause == "stalled" for d in decisions) else "budget_exhausted")
    if state.notes.get("plain_md_outcome", {}).get("ns_per_replica") != state.ns_done:
        state.notes["plain_md_outcome"] = {
            "phase": "plain_md", "reason": reason, "ns_per_replica": state.ns_done, "budget_ns": state.loop.max_ns,
            "residues": {d.residue: d.cause or d.action for d in decisions if d.action != "skip"},
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        state.save(run_dir)
    why_zh = {"target_reached": "达到目标", "stalled": "采样停滞", "budget_exhausted": "预算用完"}[reason]
    why_en = {"target_reached": "goal reached", "stalled": "sampling stalled",
              "budget_exhausted": "budget used up"}[reason]
    if not need:
        _once(run_dir, state, "plain_done",
              f"普通 MD 结束，原因是{why_zh}（每条 {state.ns_done:g} ns）：所有目标都在状态之间来回了足够多次，"
              f"比例的误差也不超过 {state.the_goal().precision:g}。结果在 {(run_dir / 'status.json').resolve()}。",
              f"Plain MD finished. Stop reason: {why_en} ({state.ns_done:g} ns per replica): every target item "
              f"crossed between states often enough, and every population error is at most "
              f"{state.the_goal().precision:g}. The results are in {(run_dir / 'status.json').resolve()}.", "done")
    elif not state.can_switch():
        _once(run_dir, state, "plain_done",
              f"普通 MD 结束，原因是{why_zh}（每条 {state.ns_done:g} ns）。这些没有采样充分：{', '.join(need)}。" + advice_zh,
              f"Plain MD finished. Stop reason: {why_en} ({state.ns_done:g} ns per replica). These are not sampled "
              f"well: {', '.join(need)}. " + advice_en, "suggestion")
    elif state.enhanced.get("enabled", True):
        _once(run_dir, state, "plain_done",
              f"普通 MD 结束，原因是{why_zh}（每条 {state.ns_done:g} ns）。这些侧链普通 MD 采样不到：{', '.join(need)}。"
              "现在自动改用增强采样（chi1 上的 metadynamics）。",
              f"Plain MD finished. Stop reason: {why_en} ({state.ns_done:g} ns per replica). Plain MD cannot sample these side chains: "
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
    from .policy import Action, record
    from .reweight import StopRule, analyze, assess, format_table
    from .reweight import save as save_populations

    files = files or load_inputs(run_dir)
    decisions = run_loop(run_dir, state, files, log)
    wanted = [d for d in decisions if d.action == "enhanced_sampling"]
    if not wanted:
        return decisions, []
    if not state.can_switch():
        log("Enhanced sampling does not start by itself: "
            + ("there is no permission to switch methods (use 'resume --enhanced-sampling' to run it)."
               if state.quantity == "chi1_populations" else "this quantity has no enhanced sampling method yet."))
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
    rule = StopRule(**{k: v for k, v in es.get("stop_rule", {}).items() if k in StopRule.__dataclass_fields__})
    rule.min_ns = min(rule.min_ns, settings.length_ns)  # A short test run may be judged at its own length.
    precision = state.the_goal().precision
    may_stop = state.allowed("stop_early")
    log(f"\nEnhanced sampling (well-tempered metadynamics on chi1) for {', '.join(labels)}: "
        f"{n_rep} replicas, up to {settings.length_ns:g} ns each, starting from the end of plain MD. Goal: every "
        f"population with a standard error of at most {precision:g}. "
        + ("It stops as soon as the goal is reached." if may_stop else
           "It runs the full length (no permission to stop early) and says when the goal is reached."))
    last = len(state.rounds_ns) - 1
    done = min(ns_done(run_dir, r) for r in range(n_rep))
    result = before = None
    verdicts: dict[str, str] = {}
    outcome = es.get("outcome") or {}
    if outcome.get("reason") == "target_reached" and may_stop:
        # Resumed after it had stopped with the goal reached: report the saved result again.
        result = analyze(run_dir, labels, n_rep, settings, state.md.temperature_k, rule)
        verdicts = assess(result, None, rule, precision) if result else {}
    else:
        while True:
            if done < settings.length_ns - 1e-9:
                target = min(done + settings.chunk_ns, settings.length_ns)
                for r in range(n_rep):  # The replicas take turns, so they can be compared after each round.
                    # A replica added later starts from the end of a plain MD replica, with its own seed.
                    start = part_path(run_dir, r % state.n_replicas, last, ".state.xml")
                    run_replica(files, run_dir, r, start, residues, settings, state.md,
                                seed=state.base_seed + 50000 + r, log=log, until_ns=target)
                done = target
            before, result = result, analyze(run_dir, labels, n_rep, settings, state.md.temperature_k, rule)
            if result is None:
                break
            verdicts = assess(result, before, rule, precision)
            save_populations(run_dir, result)
            log("\n" + format_table(result))
            reached = all(v == "reached" for v in verdicts.values())
            stalled = [k for k, v in verdicts.items() if v == "stalled"]
            if stalled and done >= rule.min_ns - 1e-9:
                _once(run_dir, state, "enhanced_stalled",
                      f"增强采样已跑每条 {done:g} ns，但 {', '.join(stalled)} 在一些重复里几乎没有在状态之间来回。"
                      "偏置势推不动它们，所以它们的比例还不能信，即使各条重复看起来一致（可能都困在同一个状态）。"
                      "常见原因是这个侧链和别的慢运动耦合在一起。",
                      f"Enhanced sampling has run {done:g} ns per replica, but {', '.join(stalled)} hardly move between "
                      "states in some replicas. The bias does not push them, so their populations cannot be trusted "
                      "yet, even if the replicas look alike (they may all be trapped in the same state). A common "
                      "cause is coupling to another slow motion.", "suggestion")
            if reached and may_stop:
                record(run_dir, state, Action("enhanced_sampling", "stop", True, "Every residue reached the goal.", done))
                break
            open_ = [k for k, v in verdicts.items() if v != "reached"]
            # Stop when every open residue stays stuck: more of the same bias is unlikely to help.
            if open_ and all(verdicts[k] == "stalled" for k in open_) and done >= rule.min_ns - 1e-9:
                es["stall_checks"] = es.get("stall_checks", 0) + 1
            else:
                es["stall_checks"] = 0
            if es["stall_checks"] >= 3 and may_stop:
                record(run_dir, state, Action("enhanced_sampling", "stop", True,
                                              f"{', '.join(open_)} stayed stuck for 3 checks in a row.", done))
                break
            # More replicas when the error comes mainly from the spread between them.
            # More replicas only when they clearly disagree, and only in the first half of the length.
            spread = [r_["residue"] for r_ in result["residues"]
                      if r_.get("verdict") == "not_yet" and r_["replicas_disagree"]]
            if spread and n_rep < state.the_goal().max_replicas and done < 0.5 * settings.length_ns:
                why = (f"The replicas clearly disagree about {', '.join(spread)}.")
                if state.allowed("add_replicas"):
                    n_rep += 1
                    settings.n_replicas = n_rep
                    es["settings"] = asdict(settings)
                    record(run_dir, state, Action("enhanced_sampling", "add_replicas", True,
                                                  why + f" Now {n_rep} replicas.", done),
                           f"{run_dir.name}：{', '.join(spread)} 的误差主要来自各条重复之间的差别。增加一条增强采样重复，"
                           f"共 {n_rep} 条；它会先补跑到每条 {done:g} ns。",
                           f"{run_dir.name}: the error of {', '.join(spread)} comes mainly from the spread between "
                           f"replicas. Adding one enhanced sampling replica, now {n_rep}; it first catches up to "
                           f"{done:g} ns.")
                else:
                    record(run_dir, state, Action("enhanced_sampling", "add_replicas", False, why, done),
                           f"{run_dir.name}：{', '.join(spread)} 的误差主要来自各条重复之间的差别，建议增加重复"
                           "（需要允许 add_replicas）。",
                           f"{run_dir.name}: the error of {', '.join(spread)} comes mainly from the spread between "
                           "replicas; I suggest more replicas (needs permission add_replicas).", "suggestion",
                           once_key="said_add_enhanced_replicas")
            if reached:
                _once(run_dir, state, "enhanced_reached",
                      f"增强采样已经达到你要的精度（每条 {done:g} ns，误差不超过 {precision:g}）。"
                      f"运行会按你的设置跑满 {settings.length_ns:g} ns；想提前结束可以说“停止 {run_dir.name}”。",
                      f"Enhanced sampling has reached the precision you asked for ({done:g} ns per replica, errors at "
                      f"most {precision:g}). The run goes on to {settings.length_ns:g} ns as you set it; say "
                      f"\"stop {run_dir.name}\" to end it now.", "info")
            if done >= settings.length_ns - 1e-9:
                # Budget used up. Run longer only with permission, only up to the limit, and only if some open
                # residue is not stuck (more time does not help a bias that does not push).
                first = es.setdefault("first_length_ns", settings.length_ns)
                cap = state.the_goal().extend_factor * first
                movable = [k for k in open_ if verdicts[k] != "stalled"]
                if not reached and movable and done < cap - 1e-9:
                    if state.allowed("extend"):
                        settings.length_ns = round(min(cap, settings.length_ns + max(settings.chunk_ns, first / 2)), 6)
                        es["settings"] = asdict(settings)
                        record(run_dir, state, Action("enhanced_sampling", "extend_budget", True,
                                                      f"{', '.join(movable)} need more data. New length "
                                                      f"{settings.length_ns:g} ns (limit {cap:g}).", done),
                               f"{run_dir.name}：增强采样的预算用完了，{', '.join(movable)} 还不够精确，但没有卡住。"
                               f"延长到每条 {settings.length_ns:g} ns（最多 {cap:g} ns）。",
                               f"{run_dir.name}: the enhanced sampling budget is used up; {', '.join(movable)} are not "
                               f"precise yet but not stuck. Extending to {settings.length_ns:g} ns per replica (at most "
                               f"{cap:g}).")
                        state.save(run_dir)
                        continue
                    record(run_dir, state, Action("enhanced_sampling", "extend_budget", False,
                                                  f"Budget used up; {', '.join(movable)} need more data.", done))
                break
        outcome = _enhanced_outcome(verdicts, done, settings.length_ns)
        es["outcome"] = outcome
        state.save(run_dir)

    folder = (run_dir / "enhanced").resolve()
    _post_outcome(run_dir, outcome, verdicts, folder, precision)
    text = (f"Enhanced sampling for {', '.join(labels)}: stop reason {outcome.get('reason', 'none')} at "
            f"{outcome.get('ns_per_replica', done):g} ns per replica. "
            + ", ".join(f"{k}: {v}" for k, v in verdicts.items()) + f". Output in {folder}.")
    log(text)
    with open(run_dir / "decisions.log", "a") as f:
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + text + "\n\n")
    return decisions, labels


STOP_REASONS = ("target_reached", "stalled", "budget_exhausted", "failed")


def _enhanced_outcome(verdicts: dict[str, str], done: float, length: float) -> dict:
    """Why enhanced sampling stopped. A stable result alone is never a reason (see reweight.assess)."""
    if verdicts and all(v == "reached" for v in verdicts.values()):
        reason = "target_reached"
    elif any(v == "stalled" for v in verdicts.values()):
        reason = "stalled"
    else:
        reason = "budget_exhausted"
    return {"phase": "enhanced_sampling", "reason": reason, "ns_per_replica": done, "budget_ns": length,
            "residues": dict(verdicts), "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def _post_outcome(run_dir: Path, outcome: dict, verdicts: dict[str, str], folder: Path, precision: float) -> None:
    reason, ns = outcome.get("reason"), outcome.get("ns_per_replica", 0)
    open_ = {k: v for k, v in verdicts.items() if v != "reached"}
    names = ", ".join(f"{k}（{v}）" for k, v in open_.items())
    names_en = ", ".join(f"{k} ({v})" for k, v in open_.items())
    where_zh = f"结果（各状态比例、误差、每条重复的翻转次数和判断）在 {folder}/populations.json，日志末尾也有表。"
    where_en = (f"The results (populations, errors, transitions per replica and verdicts) are in "
                f"{folder}/populations.json and in a table at the end of the log.")
    if reason == "target_reached":
        post(run_dir, f"结果已经可以用了：停下的原因是达到目标。每条跑了 {ns:g} ns，所有侧链的比例误差都不超过 "
                      f"{precision:g}，各条重复一致，并且都在状态之间来回了足够多次。" + where_zh,
             f"Results are ready. Stop reason: goal reached. After {ns:g} ns per replica every population has an "
             f"error of at most {precision:g}, the replicas agree, and each crossed between states often enough. "
             + where_en, "done")
    elif reason == "stalled":
        post(run_dir, f"增强采样停下了，原因是采样停滞（每条 {ns:g} ns）：{names}。stalled 表示偏置势推不动这个侧链，"
                      "它的比例不能信；not_yet 表示还不够精确；unreliable 表示结果只靠少数几帧。" + where_zh +
                      "可以加长增强采样，或者对停滞的侧链换一种方法（例如同时偏置和它耦合的运动）。",
             f"Enhanced sampling stopped. Stop reason: sampling stalled ({ns:g} ns per replica): {names_en}. stalled: "
             "the bias does not push this side chain, so its populations cannot be trusted; not_yet: not precise "
             "enough yet; unreliable: the result rests on a few frames. " + where_en + " You can run longer, or "
             "use another method for the stalled side chains (for example also bias the motion they are coupled to).",
             "suggestion")
    elif reason == "budget_exhausted":
        post(run_dir, f"增强采样停下了，原因是预算用完（每条 {ns:g} ns），还没达到目标：{names}。" + where_zh +
                      f"要达到误差 {precision:g}，可以加长，例如说“把 {run_dir.name} 的增强采样加长到 {2 * ns:g} ns”。",
             f"Enhanced sampling stopped. Stop reason: budget used up ({ns:g} ns per replica) before the goal: "
             f"{names_en}. " + where_en + f" To reach an error of {precision:g}, run longer, for example \"run the "
             f"enhanced sampling of {run_dir.name} to {2 * ns:g} ns\".", "suggestion")


def record_failure(run_dir: Path, error: BaseException) -> None:
    """Stop reason "failed": keep it in run.json and tell the chat."""
    try:
        state = RunState.load(run_dir)
        state.notes["outcome"] = {"phase": "any", "reason": "failed", "error": f"{type(error).__name__}: {error}",
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        state.save(run_dir)
    except (OSError, ValueError, TypeError, KeyError):
        pass
    post(run_dir, f"运行出错停下了，停下的原因是运行失败：{type(error).__name__}: {error}。日志里有完整的报错。"
                  f"修好问题后可以说“继续 {run_dir.name}”，它会从最后保存的位置接着跑。",
         f"The run stopped with an error. Stop reason: failed: {type(error).__name__}: {error}. The log has the "
         f"full error. After fixing the cause, say \"continue {run_dir.name}\"; it resumes from the last saved "
         "point.", "suggestion")
