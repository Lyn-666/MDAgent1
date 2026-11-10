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

from .diagnose import ResidueReport, diagnose_trajectories, format_report, load_run
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


def decide(reports: list[ResidueReport], ns_done: float, loop: LoopSettings) -> tuple[list[Decision], float]:
    """Return a decision for each residue, and the length of the next round (0 means stop)."""
    room = round(loop.max_ns - ns_done, 6)
    decisions = []
    for r in reports:
        if r.verdict == "no_chi1":
            decisions.append(Decision(r.label, "skip", r.reason))
        elif r.verdict == "ok":
            decisions.append(Decision(r.label, "done", f"Sampled well: {r.reason}"))
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
        for i in range(state.n_replicas):
            if part_path(run_dir, i, part, ".state.xml").exists():
                log(f"replica {i}: round {part + 1} is already complete.")
                continue
            seed = state.base_seed + 1000 * part + i
            run_segment(files, run_dir, i, part, seed, next_ns, state.md, log)
        state.rounds_ns.append(next_ns)
        state.pending_round_ns = 0.0
        state.save(run_dir)

        reports, decisions, next_ns = _diagnose_and_decide(run_dir, state)
        log("\n" + format_report(reports) + "\n")
        log(_record(run_dir, state, reports, decisions, next_ns))
    return decisions


def _diagnose_and_decide(run_dir: Path, state: RunState):
    reports = diagnose_trajectories(
        load_run(run_dir, state.target, len(state.rounds_ns)), state.target, state.ns_done,
        min_flips_per_replica=state.loop.min_flips_per_replica)
    decisions, next_ns = decide(reports, state.ns_done, state.loop)
    return reports, decisions, next_ns
