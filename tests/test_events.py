"""Goal "see a change" (quantity event_seen): stop once an item enters a state and stays there."""

import numpy as np
import pytest

from mdagent.observables import EventSpec, diagnose_events, first_entry, parse_events


def test_changes_are_read_and_checked():
    a, b = parse_events("A:111=t; A:88:OH-A:111:CG1@0.6=far")
    assert a.residue == ("A", 111) and a.state == "t" and a.label == "A:111=t"
    assert b.distance.cutoff_nm == 0.6 and b.state == "far" and b.label == "A:88:OH-A:111:CG1@0.6=far"
    assert EventSpec.parse("111=g-").residue == (None, 111)
    for bad in ["A:111", "A:111=x", "A:88:OH-A:111:CG1@0.6=t", "=t"]:
        with pytest.raises(ValueError):
            EventSpec.parse(bad)


def test_a_change_needs_to_come_from_another_state_and_to_last():
    t, other = 1, 0
    # Starts in the target state: not a change.
    assert first_entry(np.array([t] * 20), t, 5) is None
    # A short flicker (3 frames) does not count; a stay of 5 frames does, counted at its 5th frame.
    s = np.array([other] * 5 + [t] * 3 + [other] * 2 + [t] * 6)
    assert first_entry(s, t, 5) == 14
    # Frames between states (-1) keep the last state.
    s = np.array([other, other, -1, t, t, -1, t, t])
    assert first_entry(s, t, 5) == 7
    # Leaves the target state, then comes back: counts.
    s = np.array([t, t, other, other, t, t, t])
    assert first_entry(s, t, 3) == 6


def test_verdicts_seen_not_yet_and_never_changed():
    spec = parse_events("A:111=t")
    seen = [np.array([0] * 10 + [1] * 20), np.zeros(30, dtype=int)]
    r = diagnose_events([seen], spec, 3.0, 0.1, hold_ps=500)[0]          # 5 frames at 0.1 ns each.
    assert r.verdict == "ok" and r.extra["seen_ns"] == [1.5, None] and r.quantity == "event"
    r = diagnose_events([seen], spec, 3.0, 0.1, hold_ps=500, seen_in=2)[0]   # Needs both replicas.
    assert r.verdict == "not_yet"
    flicker = [np.array(([0] * 3 + [1] * 2) * 6)]
    assert diagnose_events([flicker], spec, 3.0, 0.1, hold_ps=500)[0].verdict == "not_yet"
    stuck = [np.zeros(30, dtype=int)] * 2
    assert diagnose_events([stuck], spec, 3.0, 0.1, hold_ps=500)[0].verdict == "no_flip"


def test_the_agent_stops_once_the_change_is_seen_and_suggests_otherwise():
    from mdagent.agent import LoopSettings, decide

    spec = parse_events("A:111=t")
    loop = LoopSettings(max_ns=10.0, extend_ns=2.0, no_flip_patience_ns=5.0)
    seen = diagnose_events([[np.array([0] * 10 + [1] * 20)]], spec, 3.0, 0.1, hold_ps=500)
    decisions, next_ns = decide(seen, 3.0, loop, may_switch=False, bias_available=False, may_stop=True)
    assert next_ns == 0 and decisions[0].action == "done"
    waiting = diagnose_events([[np.array(([0] * 3 + [1] * 2) * 6)]], spec, 3.0, 0.1, hold_ps=500)
    decisions, next_ns = decide(waiting, 3.0, loop, may_switch=False, bias_available=False, may_stop=True)
    assert next_ns == 2.0 and decisions[0].action == "extend"
    stuck = diagnose_events([[np.zeros(60, dtype=int)]], spec, 6.0, 0.1, hold_ps=500)
    decisions, next_ns = decide(stuck, 6.0, loop, may_switch=False, bias_available=False, may_stop=True)
    assert next_ns == 0 and decisions[0].action == "stalled" and decisions[0].cause == "stalled"


def test_command_line_reads_the_change_goal(tmp_path, capsys):
    from mdagent.cli import _goal_from_args, _target_from_args, build_parser, main

    args = build_parser().parse_args(["run", "--pdb", "x.pdb", "--out", "runs/x", "--observe", "A:111=t",
                                      "--quantity", "event", "--hold-ps", "50", "--seen-in", "2"])
    goal = _goal_from_args(args)
    assert goal.quantity == "event_seen" and goal.hold_ps == 50 and goal.seen_in == 2
    assert _target_from_args(args).residues == (("A", 111),)
    # A wrong change stops before anything is built.
    assert main(["run", "--pdb", "x.pdb", "--out", str(tmp_path / "r"), "--observe", "A:111=up"]) == 2
    assert "unknown state" in capsys.readouterr().err


@pytest.mark.slow
def test_plain_md_waiting_for_a_change(tmp_path):
    """Tiny peptide on CPU: a run that waits for a change ends with a stop reason and a report."""
    from dataclasses import asdict
    from pathlib import Path

    from mdagent.agent import Goal, LoopSettings, RunState, run_all
    from mdagent.build import SetupOptions, build_system
    from mdagent.report import write
    from mdagent.simulate import MDSettings, SystemFiles, write_inputs

    data = Path(__file__).parent / "data"
    built = build_system(data / "fragment_108-114.pdb", SetupOptions(forcefield="amber14", padding_nm=0.8))
    write_inputs(built, tmp_path)
    state = RunState(pdb="x", target_description="change", target=[("A", 111), ("A", 108), ("A", 114)],
                     n_replicas=2, base_seed=1,
                     md=MDSettings(equilibration_ps=1.0, save_interval_ps=0.5, platform="CPU"),
                     loop=LoopSettings(first_ns=0.002, extend_ns=0.002, max_ns=0.004, no_flip_patience_ns=0.003),
                     goal=asdict(Goal(quantity="event_seen", permissions=["stop_early"], hold_ps=1.0)),
                     events=["A:111=t", "A:108:CA-A:114:CA@1.0=far"])
    state.save(tmp_path)
    decisions, labels = run_all(tmp_path, state, SystemFiles(built.topology, built.positions, built.system),
                                log=lambda line: None)
    assert labels == [] and sorted(d.residue for d in decisions) == ["A:108:CA-A:114:CA@1=far", "A:111=t"]
    saved = RunState.load(tmp_path)
    assert saved.notes["plain_md_outcome"]["reason"] in ("target_reached", "stalled", "budget_exhausted")
    zh, en = write(tmp_path)
    assert "A:111=t" in en.read_text() and "What counts as seen" in en.read_text()
