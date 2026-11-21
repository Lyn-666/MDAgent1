"""Shadow mode: the new decision core (mdagent.core) decides next to the current policy, without acting.

Live: after every plain MD round and every enhanced sampling check, record() asks the core what it would
do with the same data and appends that decision to decisions.core.jsonl in the run folder. The run itself
still follows policy.py and agent.run_all. A shadow error never stops the run; it is logged and recorded.

Replay: replay() rebuilds, for a finished or stopped run, every decision point from the files on disk
(the first k rounds of plain MD, the first d ns of enhanced sampling) and asks the core at each one. This
needs no new simulation, so the step-5 runs can be compared at once. It writes decisions.core.replay.jsonl.

compare() puts the old action next to the core's choice at each decision point. The old action is read
from run.json (rounds, notes.actions, the stop reasons).

    python -m mdagent shadow --run runs/val_1UBQ_reach_chi1           # replay and compare
    python -m mdagent shadow --run runs/x --live                      # compare the live shadow records

Switch the live shadow off with the environment variable MDAGENT_SHADOW=0.

Task mapping (task_from_state): one requirement per estimand with the goal's precision; the budget
is the user's first plain MD limit x the first number of replicas, plus the enhanced sampling length x its
replicas when the run may use enhanced sampling (one shared budget, as planned for phase 3).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

from .core.ledger import Ledger
from .core.loop import decide
from .core.registry import ToolRegistry
from .core.task import Budget, Requirement, TaskSpec
from .estimators import default_registry

LIVE_FILE = "decisions.core.jsonl"
REPLAY_FILE = "decisions.core.replay.jsonl"
EPS = 1e-6


# ---------------------------------------------------------------- the run as a task

def first_replicas(state) -> int:
    """How many plain MD replicas the user asked for (the agent may have added some since)."""
    if "first_n_replicas" in state.notes:
        return int(state.notes["first_n_replicas"])
    adds = [a for a in state.notes.get("actions", []) if a["phase"] == "plain_md" and a["action"] == "add_replicas"
            and a["executed"]]
    if not adds:
        return state.n_replicas
    m = re.search(r"Now (\d+) replicas", adds[0]["reason"])
    first = int(m.group(1)) if m else state.n_replicas
    return max(1, first - 2)   # Older runs: policy rule 3 added two at a time.


def _adds(state, phase: str) -> list[dict]:
    return sorted([a for a in state.notes.get("actions", []) if a["phase"] == phase and a["action"] == "add_replicas"
                   and a["executed"]], key=lambda a: a["ns_per_replica"])


def replicas_at(state, phase: str, ns: float) -> int:
    """Replicas the current policy had when it decided at `ns` (replicas added at `ns` or later not counted).

    Each executed add_replicas action says "Now N replicas" in its reason.
    """
    adds = _adds(state, phase)
    before = [a for a in adds if a["ns_per_replica"] < ns - EPS]
    m = re.search(r"Now (\d+) replicas", before[-1]["reason"]) if before else None
    if m:
        return int(m.group(1))
    if phase == "plain_md":
        return first_replicas(state)
    from .enhanced import MetadSettings

    return min(MetadSettings.from_dict(state.enhanced.get("settings", {})).n_replicas, state.n_replicas) - len(adds)


def task_from_state(run_dir: Path, state, kinds: dict) -> TaskSpec:
    from .enhanced import MetadSettings
    from .reweight import StopRule

    goal, es = state.the_goal(), state.enhanced
    rule = StopRule(**{k: v for k, v in es.get("stop_rule", {}).items() if k in StopRule.__dataclass_fields__})
    n0 = first_replicas(state)
    pop = {"max_se": goal.precision, "min_events_per_replica": state.loop.min_flips_per_replica,
           "min_events_per_replica_biased": rule.min_transitions,
           "min_effective_samples": rule.min_effective_samples,
           "min_events_total": state.loop.min_flips_per_replica * n0}
    reqs = []
    if state.quantity == "event_seen":
        from .observables import parse_events
        hold = max(1, round(goal.hold_ps / state.md.save_interval_ps))
        for sp in parse_events(state.events):
            reqs.append(Requirement(sp.label, "event", {"target": sp.state, "hold_frames": hold, "seen_in": goal.seen_in}))
    else:
        from .backbone import BackboneSettings

        bb = BackboneSettings.from_dict(state.backbone)
        per_kind = {
            "state_populations": pop,
            "transition_rates": {"max_rel_error": goal.rate_rel_error, "min_population": bb.min_population,
                                 "dt_ns": state.md.save_interval_ps / 1000.0},
            "free_energy_surface": {"max_error_kcal": goal.fes_error_kcal, "cutoff_kcal": bb.fes_cutoff_kcal,
                                    "bins_deg": bb.bins_deg, "temperature_k": state.md.temperature_k},
        }
        reqs = [Requirement(e, kind, dict(per_kind[kind])) for e, kind in kinds.items()]
    perms = set(goal.permissions) | ({"switch_method"} if es.get("requested") else set())
    metad = state.quantity == "chi1_populations" and es.get("enabled", True)
    planned = state.notes.get("first_max_ns", state.loop.max_ns) * n0
    if metad and "switch_method" in perms:
        s = MetadSettings.from_dict(es.get("settings", {}))
        planned += es.get("first_length_ns", s.length_ns) * min(es.get("first_n_replicas", s.n_replicas), n0)
    budget = Budget(planned, n0, goal.extend_factor * planned if "extend" in perms else None,
                    max(goal.max_replicas, n0) if "add_replicas" in perms else None, "ns")
    method = "adaptive_md" if state.sampling == "adaptive" else "plain_md"
    return TaskSpec(goal.description or state.target_description, tuple(reqs), budget, method, state.mode,
                    tuple(sorted(perms)), ("metadynamics",) if metad else (), state.loop.extend_ns)


def tools_for(run_dir: Path, state, log: Callable = print, parts=None, n_replicas=None, until_ns=None,
              enhanced_replicas=None):
    from .tools_openmm import MetadTool, PlainMDTool

    plain = PlainMDTool(run_dir, state, log=log, parts=parts, n_replicas=n_replicas)
    metad = MetadTool(run_dir, state, log=log, until_ns=until_ns, n_replicas=enhanced_replicas)
    return plain, metad, ToolRegistry([plain, metad])


def _summary(rec: dict) -> str:
    d = rec["decision"]
    c = d["chosen"]
    what = f"stop ({rec['stop_reason']})" if c["kind"] == "stop" else f"{c['kind']} {c['tool']}"
    sugg = [f"{s['kind']} {s['tool']}" for s in d["suggestions"]]
    return what + (f"; would suggest: {', '.join(sugg)}" if sugg else "")


# ---------------------------------------------------------------- live

def record(run_dir: Path, state, phase: str, log: Callable[[str], None] = print, until_ns: float | None = None,
           enhanced_replicas: int | None = None) -> dict | None:
    """Ask the core at this decision point and append its decision to decisions.core.jsonl. Never raises."""
    if os.environ.get("MDAGENT_SHADOW") == "0":   # Switched off by the user.
        return None
    run_dir = Path(run_dir)
    ledger = Ledger(run_dir / LIVE_FILE)
    try:
        plain, metad, tools = tools_for(run_dir, state, log, until_ns=until_ns, enhanced_replicas=enhanced_replicas)
        task = task_from_state(run_dir, state, plain.kinds())
        history = [plain.name] + (["metadynamics"] if phase == "enhanced_sampling" else [])
        ns = until_ns if phase == "enhanced_sampling" else state.ns_done
        _, _, rec = decide(task, tools, default_registry(), ledger, history,
                           {"phase": phase, "ns_per_replica": ns, "shadow": True})
        log(f"Shadow (new decision core, not acted on), {phase} at {ns:g} ns: {_summary(rec)}.")
        return rec
    except Exception as e:   # The shadow must never stop a run.
        ledger.append({"kind": "shadow_error", "phase": phase, "error": f"{type(e).__name__}: {e}",
                       "progress": (ledger.last or {}).get("progress", {"looks": 0})})
        log(f"Shadow decision skipped ({type(e).__name__}: {e}).")
        return None


# ---------------------------------------------------------------- replay

def replay(run_dir: Path, log: Callable[[str], None] = print) -> list[dict]:
    """Ask the core at every past decision point of the run. Returns the decision records."""
    from .agent import RunState
    from .enhanced import MetadSettings, ns_done

    run_dir = Path(run_dir)
    state = RunState.load(run_dir)
    path = run_dir / REPLAY_FILE
    path.unlink(missing_ok=True)
    ledger = Ledger(path)
    plain, metad, tools = tools_for(run_dir, state, log)
    task = task_from_state(run_dir, state, plain.kinds())
    est = default_registry()
    out, ns = [], 0.0
    for k, length in enumerate(state.rounds_ns, 1):
        ns = round(ns + length, 9)
        plain.parts, plain.n_replicas = k, replicas_at(state, "plain_md", ns)
        metad.until_ns = 0.0
        _, _, rec = decide(task, tools, est, ledger, [plain.name], {"phase": "plain_md", "ns_per_replica": ns})
        out.append(rec)
        log(f"plain MD at {ns:g} ns, {plain.n_replicas} replicas: core would {_summary(rec)}")
    es = state.enhanced
    if metad.chosen() and (run_dir / "enhanced" / "replica_0_chi.npy").exists():
        s = MetadSettings.from_dict(es.get("settings", {}))
        plain.parts, plain.n_replicas = None, None
        n_all = min(s.n_replicas, state.n_replicas)
        end = min(ns_done(run_dir, r) for r in range(n_all))
        d = 0.0
        while d < end - EPS:
            d = round(min(d + s.chunk_ns, end), 9)
            metad.until_ns, metad.n_replicas = d, replicas_at(state, "enhanced_sampling", d)
            _, _, rec = decide(task, tools, est, ledger, [plain.name, "metadynamics"],
                               {"phase": "enhanced_sampling", "ns_per_replica": d})
            out.append(rec)
            log(f"enhanced sampling at {d:g} ns, {metad.n_replicas} replicas: core would {_summary(rec)}")
    return out


# ---------------------------------------------------------------- comparison

def old_action(run_dir: Path, state, phase: str, ns: float) -> tuple[str, str]:
    """What the current policy did at this decision point: (action, stop reason or '')."""
    acts = [a["action"] for a in state.notes.get("actions", [])
            if a["phase"] == phase and a["executed"] and abs(a["ns_per_replica"] - ns) < EPS]
    for kind in ("add_replicas", "extend_budget"):
        if kind in acts:
            return kind, ""
    es = state.enhanced
    if phase == "plain_md":
        if ns < state.ns_done - EPS:
            return "continue", ""
        o = state.notes.get("plain_md_outcome")
        if not o:
            return "running", ""
        if es.get("residues"):
            return "switch_method", o["reason"]
        return "stop", o["reason"]
    from .enhanced import MetadSettings, ns_done

    o = es.get("outcome")
    if o and ns >= o["ns_per_replica"] - EPS:
        return "stop", o["reason"]
    n = min(MetadSettings.from_dict(es.get("settings", {})).n_replicas, state.n_replicas)
    done = min(ns_done(Path(run_dir), r) for r in range(n))
    return ("continue", "") if o or ns < done - EPS else ("running", "")


SHORT = {"trapped_or_single_state": "trapped?", "not_yet_observed": "not seen yet", "slow_mixing": "slow",
         "replica_dependence": "replicas differ", "partial_trapping": "some replicas stuck",
         "estimator_invalid": "invalid estimate", "bias_ineffective": "bias does not push",
         "requirement_unreachable": "unreachable", "awaiting_confirmation": "confirming", "unexplained": "?"}


def _why(rec: dict) -> str:
    """The strongest hypothesis of each open estimand, short."""
    best = {}
    rank = {"high": 3, "medium": 2, "low": 1}
    for h in rec["hypotheses"]:
        if rank[h["plausibility"]] > best.get(h["estimand"], (0, ""))[0]:
            best[h["estimand"]] = (rank[h["plausibility"]], SHORT.get(h["id"], h["id"]))
    return ", ".join(f"{e}: {w}" for e, (_, w) in best.items()) or "all met"


def compare(run_dir: Path, records: list[dict]) -> tuple[str, dict]:
    """A table of old action vs core choice at each decision point, and counts.

    same: the same action (and the same stop reason); other_stop_reason: both stop, for different reasons;
    core_stops_earlier: the core would stop where the policy went on; core_goes_on: the reverse;
    different: another action.
    """
    from .agent import RunState

    state = RunState.load(Path(run_dir))
    counts = {"same": 0, "other_stop_reason": 0, "core_stops_earlier": 0, "core_goes_on": 0, "different": 0,
              "errors": 0}
    rows, first_stop = [], None
    for r in records:
        if r.get("kind") != "decision":
            counts["errors"] += r.get("kind") == "shadow_error"
            continue
        phase, ns = r["phase"], r["ns_per_replica"]
        old, old_reason = old_action(run_dir, state, phase, ns)
        c = r["decision"]["chosen"]
        core, reason = c["kind"], r["stop_reason"] or ""
        if old == "running":
            verdict = "-"
        elif core == old == "stop" and reason != old_reason:
            verdict = "other_stop_reason"
        elif core == old:
            verdict = "same"
        elif core == "stop":
            verdict = "core_stops_earlier"
        elif old == "stop":
            verdict = "core_goes_on"
        else:
            verdict = "different"
        if verdict in counts:
            counts[verdict] += 1
        if core == "stop" and first_stop is None:
            first_stop = (phase, ns, reason, r["progress"]["spent"])
        rows.append(f"{phase:<18}{ns:>9g}  {old + (' (' + old_reason + ')' if old_reason else ''):<32}"
                    f"{core + (' (' + reason + ')' if reason else ' ' + c['tool']):<34}{verdict:<20}{_why(r)}")
    head = (f"{'phase':<18}{'ns/repl':>9}  {'current policy did':<32}{'new core would':<34}{'comparison':<20}"
            "open estimands: strongest hypothesis (heuristic)")
    tail = [f"Counts: {counts}."]
    if first_stop:
        tail.append(f"The core would first stop in {first_stop[0]} at {first_stop[1]:g} ns per replica "
                    f"({first_stop[2]}), after {first_stop[3]:g} replica-ns in all.")
    return "\n".join([head, *rows, *tail]), counts


def load(path: Path) -> list[dict]:
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
