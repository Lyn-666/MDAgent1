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
from .uncertainty import Precision, planned_looks


@dataclass
class LoopSettings:
    first_ns: float = 1.0           # Length of round 1, per replica.
    extend_ns: float = 5.0          # Length of each later round, per replica.
    max_ns: float = 20.0            # Budget: total length per replica.
    # Give up on plain MD for an item with no state change after this many ns per replica. None (the default for
    # new runs): judge from the budget instead (no_change_hopeless), so no system-specific time is needed.
    no_flip_patience_ns: float | None = None
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
    backbone: list = field(default_factory=list)   # Basin specs (basins.py), e.g. "rama:A:110", "turn:A:52".
    events: list = field(default_factory=list)     # Changes to see (observables.EventSpec) for quantity event_seen.
    backbone_settings: dict = field(default_factory=dict)  # backbone.BackboneSettings fields (own basins, margins)
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

    def precision(self) -> "Precision":
        """The precision the goal asks for, with the number of decision points this budget allows."""
        g = self.the_goal()
        if g.allocation == "projection":
            # The agent may stop at no more than stop_looks looks, so its intervals are calibrated for that many.
            return Precision(epsilon=g.precision, criterion="confidence", alpha=g.alpha, looks=g.stop_looks)
        factor = g.extend_factor if "extend" in g.permissions else 1.0
        return Precision(epsilon=g.precision, criterion=g.criterion, alpha=g.alpha,
                         looks=planned_looks(self.loop.first_ns, self.loop.extend_ns, self.loop.max_ns, factor))

    def max_ns_per_replica(self) -> float:
        """Longest a replica may run with projection allocation: the user's max_ns, never more (user's decision,
        2026-11-20: the time and replica numbers the user sets are limits)."""
        return self.loop.max_ns

    def budget_ns(self) -> float:
        """Production ns of all replicas together that the agent may spend in plain MD, never more. Default: the
        limits the user set, max_ns per replica times the most replicas the agent may run (max_replicas when it
        may add replicas, otherwise the replicas it starts with)."""
        g = self.the_goal()
        if g.budget_ns > 0:
            return g.budget_ns
        most = g.max_replicas if self.allowed("add_replicas") else self.n_replicas_at_start()
        return most * self.loop.max_ns

    def n_replicas_at_start(self) -> int:
        """Replicas the user asked for (the agent may have added some since)."""
        n = self.notes.get("replicas_at_start", self.notes.get("first_n_replicas"))
        return int(n if n is not None else self.n_replicas)

    def can_switch(self) -> bool:
        """May the agent switch to enhanced sampling by itself? Needs the permission, and a quantity that
        enhanced sampling can push (only chi1 for now)."""
        return self.allowed("switch_method") and self.quantity in BIASABLE

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
        if isinstance(d.get("backbone"), dict):
            # Runs from before the two backbone implementations were merged (2026-11-22) kept the basin settings
            # in "backbone" and measured every target residue: they become rama items of those residues.
            d["backbone_settings"] = d.pop("backbone")
            if (d.get("goal") or {}).get("quantity") == "backbone_basins":
                d["backbone"] = [f"rama:{c}:{n}" if c else f"rama:{n}" for c, n in d["target"]]
        elif d.get("backbone") and "backbone_settings" not in d:
            # rama: runs from the other branch before the merge used three centres: keep them, so a resumed run
            # does not change its basins halfway.
            d["backbone_settings"] = {"rama": "centres"}
        return cls(**d)


PERMISSIONS = ("extend", "add_replicas", "switch_method", "stop_early")
# Quantities with an enhanced sampling method (enhanced.py): what metadynamics biases for each.
BIASABLE = {"chi1_populations": "chi1", "backbone_basins": "backbone"}
BIAS_TEXT = {"chi1": ("chi1 上的 metadynamics", "metadynamics on chi1"),
             "backbone": ("φ 和 ψ 上的 metadynamics", "metadynamics on phi and psi")}
DEFAULT_PERMISSIONS = {
    # auto: the user's time and replica numbers are limits, and changing the method needs the user's yes
    # (decided 2026-11-20). The agent stops early when the goal is reached or when going on would not
    # help, and suggests the rest (run longer, more replicas, another method) in the chat.
    "auto": ["stop_early"],
    # guided: the user chose the method and the budget; the agent runs it all and only suggests.
    "guided": [],
}


@dataclass
class Goal:
    """What the user wants to know, how precisely, and what the agent may do by itself.

    quantity: the result the user cares about. Supported now: "chi1_populations" (the state
      populations of the target side chains).
    precision: the tolerance on any state population (0 to 1).
    allocation: "projection" lets the agent choose replicas, lengths and the stop for the least
      compute (allocation.py), always inside the user's limits; "rounds" keeps the fixed rounds of
      earlier versions.
    criterion: what has to be at most `precision`. "standard_error" (the default, and what
      every run before this field meant), or "confidence" / "sequence", which widen the
      interval because the agent judges the same growing data many times before it stops
      (uncertainty.Precision). The budget comparison (budget.py) shows what each costs.
    permissions: actions the agent may take without asking (default in auto mode: only stop_early;
      the others become suggestions that the user confirms):
      extend         run longer than the planned budget when the goal is not met yet
      add_replicas   start more independent replicas than the user asked for
      switch_method  switch from plain MD to enhanced sampling
      stop_early     stop before the budget once the goal is met, or once going on would not help
    """

    quantity: str = "chi1_populations"   # or "distance_states", "event_seen", "backbone_basins" (observables.py)
    description: str = ""          # The user's own words.
    precision: float = 0.05
    criterion: str = "standard_error"  # How precision is measured: see uncertainty.Precision.
    alpha: float = 0.05               # Allowed chance of being outside the interval.
    permissions: list = field(default_factory=lambda: list(PERMISSIONS))
    max_replicas: int = 10          # add_replicas stops here (literature: 5 to 10 replicas).
    extend_factor: float = 2.0      # extend may run to this many times the first budget.
    # How time is spent (allocation.py). "rounds": every round adds extend_ns to every replica, as before.
    # "projection": the agent decides how many replicas, how long and when to stop, for the least compute;
    # it looks at most stop_looks times (after one planning look) and its intervals are calibrated for that.
    allocation: str = "rounds"
    budget_ns: float = 0.0          # production ns of all replicas together; 0: replicas x max_ns
    stop_looks: int = 3
    hold_ps: float = 100.0          # event_seen: the new state must last this long to count as seen.
    seen_in: int = 1                # event_seen: in how many replicas the change must be seen.
    rate_rel_error: float = 0.25    # backbone_basins: largest relative error of the leading transition rates.
    fes_error_kcal: float = 0.5     # backbone_basins: largest error of the phi/psi free energy where it is low.

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


def no_change_hopeless(report: ResidueReport, ns_done: float, loop: LoopSettings, precision=None) -> bool:
    """Has an item with no state change at all been watched long enough to give up on plain MD for it?

    With a patience time (older runs, or --no-flip-patience-ns): after that time. Otherwise from the data and the
    budget (uncertainty.never_changing_hopeless): even at the fastest rate that zero changes allow, the rest of the
    budget would not give the changes the goal needs (uncertainty.changes_needed: enough in every replica, and
    enough for the precision when it is known). The projection allocation uses the same rule.
    """
    from .uncertainty import Precision, changes_needed, never_changing_hopeless

    if loop.no_flip_patience_ns is not None:
        return ns_done >= loop.no_flip_patience_ns
    n = max(len(report.transitions), 1)
    p = Precision.of(precision)
    need = changes_needed(n, loop.min_flips_per_replica, p.epsilon if p else None, p.z if p else 1.0)
    return never_changing_hopeless(ns_done * n, (loop.max_ns - ns_done) * n, need)


def decide(reports: list[ResidueReport], ns_done: float, loop: LoopSettings,
           may_switch: bool = True, bias_available: bool = True,
           may_stop: bool = False, precision=None) -> tuple[list[Decision], float]:
    """Return a decision for each item (residue or distance), and the length of the next round (0: stop).

    An item with no state change after the patience time is closed for plain MD (more plain MD would not
    help) when the agent may switch methods or may stop early. Otherwise (guided mode: the user chose plain
    MD and its length) it goes on until the budget, marked stalled.
    Without an enhanced sampling method for the quantity, such an item gets the action "stalled"
    instead of "enhanced_sampling" when plain MD stops.
    """
    need = "enhanced_sampling" if bias_available else "stalled"
    room = round(loop.max_ns - ns_done, 6)
    decisions = []
    for r in reports:
        if r.verdict in ("no_chi1", "fixed_chi1", "no_phi_psi"):
            decisions.append(Decision(r.label, "skip", r.reason))
        elif r.verdict == "ok":
            decisions.append(Decision(r.label, "done", f"Sampled well: {r.reason}"))
        elif sum(r.transitions) == 0 and no_change_hopeless(r, ns_done, loop, precision) \
                and not (may_switch and bias_available) and not may_stop and room > 0:
            decisions.append(Decision(
                r.label, "extend",
                f"No flip in {ns_done:g} ns per replica. Plain MD is unlikely to see one, but the user chose "
                "plain MD, so it goes on until the budget.", stalled=True))
        elif sum(r.transitions) == 0 and no_change_hopeless(r, ns_done, loop, precision):
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
    projection = state.the_goal().allocation == "projection"
    if projection:
        # What the agent looked at: each item's verdict, largest interval half-width and state changes.
        z = state.precision().z
        for r in reports:
            if r.standard_error:
                lines.append(f"  {r.label}: {r.verdict}; largest half-width {z * max(r.standard_error.values()):.3f} "
                             f"(asked {state.the_goal().precision:g}); state changes per replica {r.transitions}.")
    else:
        for d in decisions:
            if d.action in ("extend", "enhanced_sampling", "stalled"):
                lines.append(f"  {d.residue}: {d.action}. {d.reason}")
        n_done = sum(d.action == "done" for d in decisions)
        if n_done:
            lines.append(f"  {n_done} item(s) are sampled well.")
    acts = [a for a in state.notes.get("actions", []) if a.get("phase") == "plain_md"]
    if projection and acts and acts[-1].get("ns_per_replica") == state.ns_done:
        lines.append(f"Agent: {acts[-1]['reason']}")
    elif next_ns > 0:
        lines.append(f"Decision: run every replica {next_ns:g} ns longer.")
    if next_ns <= 0:
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
        round_started = time.time()
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
        # Measured cost of this round, so the cost model can be checked against the clock (cost.py).
        state.notes.setdefault("rounds", []).append(
            {"ns": round(next_ns, 6), "n_replicas": state.n_replicas,
             "seconds": round(time.time() - round_started, 1)})
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


def _decision_loop(state: RunState) -> LoopSettings:
    """The limits decide() judges against: the same ones the projection allocation uses."""
    from dataclasses import replace

    if state.the_goal().allocation == "projection":
        return replace(state.loop, max_ns=state.max_ns_per_replica())
    return state.loop


def diagnose_run(run_dir: Path, state: RunState) -> list[ResidueReport]:
    """Diagnose the plain MD of a run for its quantity. The run and `mdagent diagnose` both use this, so
    they always judge the same way."""
    if state.quantity == "distance_states":
        return _diagnose_distances(run_dir, state)
    if state.quantity == "event_seen":
        return _diagnose_events(run_dir, state)
    if state.quantity == "backbone_basins":
        return _diagnose_backbone(run_dir, state)
    fixed = fixed_chi1_residues(run_dir / "protein.pdb")
    if state.sampling == "adaptive":
        # Segments start from different frames on purpose, so judge the pooled flips.
        return diagnose_trajectories(
            load_run(run_dir, state.target, len(state.rounds_ns), join=False), state.target,
            state.ns_done, min_flips_per_replica=state.loop.min_flips_per_replica,
            pooled_total_ns=state.ns_done * state.n_replicas, n_replicas=state.n_replicas, fixed=fixed,
            precision=state.precision())
    return diagnose_trajectories(
        load_run(run_dir, state.target, len(state.rounds_ns)), state.target, state.ns_done,
        min_flips_per_replica=state.loop.min_flips_per_replica, fixed=fixed,
        precision=state.precision())


def _diagnose_and_decide(run_dir: Path, state: RunState):
    reports = diagnose_run(run_dir, state)
    decisions, next_ns = decide(reports, state.ns_done, _decision_loop(state), state.allowed("switch_method"),
                                bias_available=state.quantity in BIASABLE,
                                may_stop=state.allowed("stop_early"), precision=state.precision())
    next_ns = _apply_policy(run_dir, state, reports, decisions, next_ns)
    _post_plain_md_messages(run_dir, state, decisions, next_ns)
    return reports, decisions, next_ns


def _apply_policy(run_dir: Path, state: RunState, reports, decisions, next_ns: float) -> float:
    """Choose the next action once per round (policy.py for rounds, allocation.py for projection);
    a resumed run keeps the choice it made."""
    from .policy import allocation_next, plain_md_next

    key = {"ns": state.ns_done, "max_ns": state.loop.max_ns, "n_replicas": state.n_replicas,
           "permissions": sorted(state.the_goal().permissions)}
    if state.the_goal().allocation == "projection":
        from .allocation import RULE_VERSION

        key["rule"] = RULE_VERSION   # A newer rule decides again when a run is resumed.
    done = state.notes.get("policy_round")
    if done and done.get("key") == key:
        if done.get("decisions"):   # Projection: restore what it decided for each item.
            for d in decisions:
                if d.residue in done["decisions"]:
                    d.action, d.cause = done["decisions"][d.residue]
        return done["next_ns"]  # Same round, same budget: keep the choice (no second message, no second replica).
    if state.the_goal().allocation == "projection":
        next_ns = allocation_next(run_dir, state, reports, decisions)
        # Keyed as the state is left after the choice (more replicas), which is what a resumed run sees.
        state.notes["policy_round"] = {"key": {**key, "max_ns": state.loop.max_ns, "n_replicas": state.n_replicas},
                                       "next_ns": next_ns,
                                       "decisions": {d.residue: [d.action, d.cause] for d in decisions}}
        state.save(run_dir)
        return next_ns
    next_ns = plain_md_next(run_dir, state, reports, decisions, next_ns)
    key.update(max_ns=state.loop.max_ns, n_replicas=state.n_replicas)  # As the policy left them.
    state.notes["policy_round"] = {"key": key, "next_ns": next_ns}
    state.save(run_dir)
    from .shadow import record as shadow_record

    shadow_record(run_dir, state, "plain_md")   # What the new decision core would do; not acted on.
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
                              state.loop.min_flips_per_replica, state.precision())


def _diagnose_backbone(run_dir: Path, state: RunState) -> list[ResidueReport]:
    """Backbone items (rama:A:110, turn:A:52): basin populations, transition statistics and the free-energy
    landscape of each, judged together (backbone.py); the details are written to backbone/."""
    from .backbone import BackboneSettings, diagnose_backbone, save_details
    from .tools_openmm import backbone_angles

    settings = BackboneSettings.from_dict(state.backbone_settings)
    items, skipped = backbone_angles(run_dir, state, len(state.rounds_ns))
    goal = state.the_goal()
    reports, details = diagnose_backbone(items, settings, state.ns_done, state.md.save_interval_ps / 1000.0,
                                         state.md.temperature_k, state.loop.min_flips_per_replica, state.precision(),
                                         goal.rate_rel_error, goal.fes_error_kcal)
    save_details(run_dir / "backbone", details)
    for label, why in skipped:
        reports.append(ResidueReport(chain="", resseq=0, resname="", name=label, transitions=[], occupancies=[],
                                     dominant=[], quantity="backbone", verdict="no_phi_psi",
                                     reason=f"Not measured: {why}"))
    return reports


def _diagnose_events(run_dir: Path, state: RunState) -> list[ResidueReport]:
    """Changes to see: per-frame states of each item (chi1 or distance), then observables.diagnose_events."""
    from .observables import diagnose_events, parse_events
    from .tools_openmm import event_series

    # Adaptive sampling: each segment starts from a chosen frame, so judge every segment on its own. Joined,
    # a segment that starts in the target state would look like an entry into it.
    series = event_series(run_dir, state, len(state.rounds_ns), join=state.sampling != "adaptive")
    goal = state.the_goal()
    return diagnose_events(series, parse_events(state.events), state.ns_done, state.md.save_interval_ps / 1000.0,
                           hold_ps=goal.hold_ps, seen_in=goal.seen_in)


def _once(run_dir: Path, state: RunState, key: str, zh: str, en: str, kind: str) -> None:
    """Post a message to the chat, but only once per run."""
    if state.notes.get(key):
        return
    post(run_dir, zh, en, kind)
    state.notes[key] = True
    state.save(run_dir)


def _advice(state: RunState, run_dir: Path, decisions: list[Decision]) -> tuple[str, str]:
    """What to try next for the items plain MD did not sample, by cause (Chinese, English). Only suggestions:
    the user confirms any change of method, length or number of replicas."""
    name = run_dir.name
    open_ = [d for d in decisions if d.action in ("enhanced_sampling", "stalled", "extend")]
    never = [d.residue for d in open_ if d.cause == "stalled" or d.stalled]
    slow = [d.residue for d in open_ if d.residue not in never]
    zh, en = [], []
    if slow:
        zh.append(f"{', '.join(slow)} 在状态之间有来回，只是太慢或各重复不一致：建议用自适应采样再跑（从访问少的状态的帧重新起跑），"
                  f"可以说“用自适应采样重新跑 {name}”；也可以提高上限，例如说“把 {name} 加长到 … ns”。")
        en.append(f"{', '.join(slow)} change state, but too slowly or differently between replicas: I suggest adaptive "
                  f"sampling (restart from frames of rarely visited states), say \"run {name} again with adaptive "
                  f"sampling\"; or raise the limit, for example \"run {name} to … ns\".")
    if never and state.quantity in BIASABLE:
        zh.append(f"{', '.join(never)} 一次都没有变化：自适应采样只能从已经见过的状态出发，帮不上忙。建议用增强采样，"
                  f"可以说“用增强采样继续 {name}”。")
        en.append(f"{', '.join(never)} never changed state: adaptive sampling can only restart from states already "
                  f"seen, so it cannot help. I suggest enhanced sampling: say \"continue {name} with enhanced "
                  "sampling\".")
        if state.quantity == "backbone_basins":
            zh.append("增强采样能给出构象区比例和自由能面，但会改变动力学：转变速率和停留时间仍然只能来自普通 MD，"
                      "需要普通 MD 里看到转变（可以提高上限）。")
            en.append("Enhanced sampling gives the basin populations and the landscape, but it changes the kinetics: "
                      "transition rates and dwell times still come from plain MD only and need changes seen there "
                      "(you can raise the limit).")
    elif never:
        zh.append(f"{', '.join(never)} 一次都没有变化，这个量还没有增强采样方法。它也可能只有一个稳定状态。")
        en.append(f"{', '.join(never)} never changed state, and this quantity has no enhanced sampling method yet. "
                  "It may also have only one stable state.")
    zh.append("这些都要你确认后才会做。")
    en.append("Nothing of this is done without your confirmation.")
    return "".join(zh), " ".join(en)


def _post_plain_md_messages(run_dir: Path, state: RunState, decisions: list[Decision], next_ns: float) -> None:
    # With projection allocation the agent has already explained its choice (policy.allocation_next).
    projection = state.the_goal().allocation == "projection"
    stalled = [] if projection else [d.residue for d in decisions if d.stalled]
    advice_zh, advice_en = _advice(state, run_dir, decisions)
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
    # Why plain MD stopped. Stalled and not_reachable win over budget: they say that more plain MD would not help.
    reason = ("target_reached" if not need else
              "stalled" if any(d.cause == "stalled" for d in decisions) else
              "insufficient_evidence" if any(d.cause == "insufficient_evidence" for d in decisions) else
              "not_reachable" if any(d.cause == "not_reachable" for d in decisions) else "budget_exhausted")
    if state.notes.get("plain_md_outcome", {}).get("ns_per_replica") != state.ns_done:
        state.notes["plain_md_outcome"] = {
            "phase": "plain_md", "reason": reason, "ns_per_replica": state.ns_done, "budget_ns": state.loop.max_ns,
            "residues": {d.residue: d.cause or d.action for d in decisions if d.action != "skip"},
            "decision": (state.notes.get("allocation_decision") or {}).get("decision",
                                                                           "STOP" if not need else "INSUFFICIENT_EVIDENCE"),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        state.save(run_dir)
    why_zh = {"target_reached": "达到目标", "stalled": "采样停滞", "budget_exhausted": "预算用完",
              "not_reachable": "上限内达不到", "insufficient_evidence": "证据不足"}[reason]
    why_en = {"target_reached": "goal reached", "stalled": "sampling stalled", "budget_exhausted": "budget used up",
              "not_reachable": "not reachable within the limit", "insufficient_evidence": "insufficient evidence"}[reason]
    if not need and state.quantity == "event_seen":
        seen = "；".join(d.reason.replace("Sampled well: ", "") for d in decisions if d.action == "done")
        _once(run_dir, state, "plain_done",
              f"看到了你要的变化，普通 MD 提前停下（每条 {state.ns_done:g} ns，上限 {state.loop.max_ns:g} ns）。{seen}"
              f"轨迹和结果在 {run_dir.resolve()}。",
              f"The change you asked for was seen, so plain MD stopped early ({state.ns_done:g} ns per replica, "
              f"limit {state.loop.max_ns:g} ns). {seen} Trajectories and results are in {run_dir.resolve()}.", "done")
    elif not need:
        _once(run_dir, state, "plain_done",
              f"普通 MD 结束，原因是{why_zh}（每条 {state.ns_done:g} ns）：所有目标都在状态之间来回了足够多次，"
              f"{state.precision().describe(zh=True)}。结果在 {(run_dir / 'status.json').resolve()}。",
              f"Plain MD finished. Stop reason: {why_en} ({state.ns_done:g} ns per replica): every target item "
              f"crossed between states often enough, with {state.precision().describe()}. "
              f"The results are in {(run_dir / 'status.json').resolve()}.", "done")
    elif not state.can_switch():
        _once(run_dir, state, "plain_done",
              f"普通 MD 结束，原因是{why_zh}（每条 {state.ns_done:g} ns）。这些没有采样充分：{', '.join(need)}。" + advice_zh,
              f"Plain MD finished. Stop reason: {why_en} ({state.ns_done:g} ns per replica). These are not sampled "
              f"well: {', '.join(need)}. " + advice_en, "suggestion")
    elif state.enhanced.get("enabled", True):
        _once(run_dir, state, "plain_done",
              f"普通 MD 结束，原因是{why_zh}（每条 {state.ns_done:g} ns）。这些侧链普通 MD 采样不到：{', '.join(need)}。"
              f"现在自动改用增强采样（{BIAS_TEXT[BIASABLE[state.quantity]][0]}）。",
              f"Plain MD finished. Stop reason: {why_en} ({state.ns_done:g} ns per replica). Plain MD cannot sample these side chains: "
              f"{', '.join(need)}. Switching to enhanced sampling ({BIAS_TEXT[BIASABLE[state.quantity]][1]}) now.", "info")


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
               if state.quantity in BIASABLE else "this quantity has no enhanced sampling method yet."))
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
        if state.quantity == "backbone_basins":
            es["items"] = [d.residue for d in chosen]   # The item specs (rama:A:110, turn:A:52) to bias.
    settings = MetadSettings.from_dict(es.get("settings", {}))
    es["settings"] = asdict(settings)
    n_rep = min(settings.n_replicas, state.n_replicas)
    state.save(run_dir)

    if state.quantity == "backbone_basins":
        from .backbone import sets_of

        labels = list(es["items"])
        by_label = {s.label: s for s in sets_of(state)}
        bias_what, item_sets = labels, [by_label[k] for k in labels]
    else:
        labels = [f"{es['resnames'][f'{c}:{n}']}{n}" for c, n in residues]
        bias_what, item_sets = residues, None
    rule = StopRule(**{k: v for k, v in es.get("stop_rule", {}).items() if k in StopRule.__dataclass_fields__})
    rule.min_ns = min(rule.min_ns, settings.length_ns)  # A short test run may be judged at its own length.
    precision = state.precision()
    may_stop = state.allowed("stop_early")
    kind = BIASABLE[state.quantity]
    goal = state.the_goal()
    fes_error = goal.fes_error_kcal if kind == "backbone" else None
    log(f"\nEnhanced sampling (well-tempered {BIAS_TEXT[kind][1]}) for {', '.join(labels)}: "
        f"{n_rep} replicas, up to {settings.length_ns:g} ns each, starting from the end of plain MD. Goal: every "
        f"{precision.describe()}"
        + (f", and the reweighted phi/psi landscape with an error of at most {fes_error:g} kcal/mol where it is low "
           "(transition rates come from plain MD only: the bias changes the kinetics)" if kind == "backbone" else "")
        + ". "
        + ("It stops as soon as the goal is reached." if may_stop else
           "It runs the full length (no permission to stop early) and says when the goal is reached."))
    last = len(state.rounds_ns) - 1
    done = min(ns_done(run_dir, r) for r in range(n_rep))
    result = before = None
    verdicts: dict[str, str] = {}
    outcome = es.get("outcome") or {}
    if outcome.get("reason") == "target_reached" and may_stop:
        # Resumed after it had stopped with the goal reached: report the saved result again.
        result = analyze(run_dir, labels, n_rep, settings, state.md.temperature_k, rule, kind, item_sets)
        verdicts = assess(result, None, rule, precision, fes_error) if result else {}
    else:
        while True:
            if done < settings.length_ns - 1e-9:
                target = min(done + settings.chunk_ns, settings.length_ns)
                for r in range(n_rep):  # The replicas take turns, so they can be compared after each round.
                    # A replica added later starts from the end of a plain MD replica, with its own seed.
                    start = part_path(run_dir, r % state.n_replicas, last, ".state.xml")
                    run_replica(files, run_dir, r, start, bias_what, settings, state.md,
                                seed=state.base_seed + 50000 + r, log=log, until_ns=target, kind=kind)
                done = target
            before, result = result, analyze(run_dir, labels, n_rep, settings, state.md.temperature_k, rule, kind,
                                             item_sets)
            if result is None:
                break
            verdicts = assess(result, before, rule, precision, fes_error)
            save_populations(run_dir, result)
            log("\n" + format_table(result))
            from .shadow import record as shadow_record

            shadow_record(run_dir, state, "enhanced_sampling", log, until_ns=done, enhanced_replicas=n_rep)
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
                    es.setdefault("first_n_replicas", n_rep)   # Enhanced sampling replicas the run started with.
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
                      f"增强采样已经达到你要的精度（每条 {done:g} ns，{precision.describe(zh=True)}）。"
                      f"运行会按你的设置跑满 {settings.length_ns:g} ns；想提前结束可以说“停止 {run_dir.name}”。",
                      f"Enhanced sampling has reached the precision you asked for ({done:g} ns per replica, errors at "
                      f"{precision.describe()}). The run goes on to {settings.length_ns:g} ns as you set it; say "
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


STOP_REASONS = ("target_reached", "stalled", "not_reachable", "budget_exhausted", "insufficient_evidence", "failed")


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


def _post_outcome(run_dir: Path, outcome: dict, verdicts: dict[str, str], folder: Path,
                  precision: Precision) -> None:
    reason, ns = outcome.get("reason"), outcome.get("ns_per_replica", 0)
    open_ = {k: v for k, v in verdicts.items() if v != "reached"}
    names = ", ".join(f"{k}（{v}）" for k, v in open_.items())
    names_en = ", ".join(f"{k} ({v})" for k, v in open_.items())
    where_zh = f"结果（各状态比例、误差、每条重复的翻转次数和判断）在 {folder}/populations.json，日志末尾也有表。"
    where_en = (f"The results (populations, errors, transitions per replica and verdicts) are in "
                f"{folder}/populations.json and in a table at the end of the log.")
    if reason == "target_reached":
        post(run_dir, f"结果已经可以用了：停下的原因是达到目标。每条跑了 {ns:g} ns，所有侧链的比例误差都不超过 "
                      f"{precision.epsilon:g}，各条重复一致，并且都在状态之间来回了足够多次。" + where_zh,
             f"Results are ready. Stop reason: goal reached. After {ns:g} ns per replica every population has an "
             f"error of at most {precision.epsilon:g}, the replicas agree, and each crossed between states often enough. "
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
                      f"要达到误差 {precision.epsilon:g}，可以加长，例如说“把 {run_dir.name} 的增强采样加长到 {2 * ns:g} ns”。",
             f"Enhanced sampling stopped. Stop reason: budget used up ({ns:g} ns per replica) before the goal: "
             f"{names_en}. " + where_en + f" To reach an error of {precision.epsilon:g}, run longer, for example \"run the "
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
