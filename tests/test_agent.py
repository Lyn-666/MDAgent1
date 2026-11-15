from pathlib import Path

import pytest

from mdagent.agent import LoopSettings, decide
from mdagent.diagnose import ResidueReport, judge, rank

DATA = Path(__file__).resolve().parent / "data"


def _report(resseq, transitions, dominant, ns=1.0):
    occ = [{"g+": 0.0, "t": 0.0, "g-": 0.0} for _ in dominant]
    for o, d in zip(occ, dominant):
        o[d] = 1.0
    r = ResidueReport("A", resseq, "VAL", transitions, occ, dominant, ns_per_replica=ns)
    judge(r)
    return r


def test_rank_puts_slowest_first_and_ok_last():
    fast = _report(1, [9, 8, 9], ["t", "t", "t"])          # ok
    one_flip = _report(2, [1, 0, 0], ["t", "g-", "g-"])    # disagree, ~3 ns between flips
    frozen = _report(3, [0, 0, 0], ["t", "t", "t"])        # no flip, > 3 ns
    some = _report(4, [2, 1, 3], ["t", "t", "t"])          # rare flips, ~0.5 ns
    ranked = rank([fast, some, frozen, one_flip])
    assert [r.resseq for r in ranked] == [2, 3, 4, 1]
    # Same estimated time (3 ns): the residue whose replicas disagree comes first.
    assert ranked[0].disagreement == 1.0
    assert frozen.time_between_flips_text() == ">3.0 ns"
    assert some.time_between_flips_text() == "~0.5 ns"


def test_decide_extends_while_plain_md_is_useful():
    loop = LoopSettings(extend_ns=5, max_ns=20, no_flip_patience_ns=10)
    reports = [
        _report(1, [9, 8, 9], ["t", "t", "t"]),
        _report(2, [0, 0, 0], ["t", "t", "t"]),
        _report(3, [1, 0, 0], ["t", "g-", "g-"]),
    ]
    decisions, next_ns = decide(reports, ns_done=1.0, loop=loop)
    assert [d.action for d in decisions] == ["done", "extend", "extend"]
    assert next_ns == 5


def test_decide_gives_up_on_frozen_side_chain_after_patience():
    loop = LoopSettings(extend_ns=5, max_ns=20, no_flip_patience_ns=10)
    reports = [_report(1, [0, 0, 0], ["t", "t", "t"], ns=11)]
    decisions, next_ns = decide(reports, ns_done=11.0, loop=loop)
    assert decisions[0].action == "enhanced_sampling"
    assert next_ns == 0


def test_replicas_in_different_states_without_flips_is_disagreement():
    r = _report(1, [0, 0], ["g+", "t"])
    assert r.verdict == "replicas_disagree"
    loop = LoopSettings(extend_ns=5, max_ns=20, no_flip_patience_ns=10)
    assert decide([r], ns_done=1.0, loop=loop)[0][0].action == "extend"
    r = _report(1, [0, 0], ["g+", "t"], ns=12)
    assert decide([r], ns_done=12.0, loop=loop)[0][0].action == "enhanced_sampling"


def test_time_text_uses_ps_for_short_runs():
    assert _report(1, [0, 0], ["t", "t"], ns=0.004).time_between_flips_text() == ">8 ps"


def test_decide_respects_budget():
    loop = LoopSettings(extend_ns=5, max_ns=8, no_flip_patience_ns=100)
    reports = [_report(1, [1, 0, 0], ["t", "g-", "g-"], ns=6)]
    decisions, next_ns = decide(reports, ns_done=6.0, loop=loop)
    assert decisions[0].action == "extend" and next_ns == 2  # Only 2 ns left in the budget.
    decisions, next_ns = decide(reports, ns_done=8.0, loop=loop)
    assert decisions[0].action == "enhanced_sampling" and next_ns == 0


@pytest.mark.slow
def test_run_resume_and_finish_interrupted_round(tmp_path):
    """A tiny peptide on CPU: run, extend the budget, and recover from a stopped round."""
    from mdagent.agent import RunState, run_loop
    from mdagent.build import SetupOptions, build_system
    from mdagent.diagnose import load_run
    from mdagent.simulate import MDSettings, SystemFiles, write_inputs

    pdb = DATA / "fragment_108-114.pdb"
    built = build_system(pdb, SetupOptions(forcefield="amber14", padding_nm=0.8))  # Amber is faster on CPU.
    write_inputs(built, tmp_path)
    state = RunState(
        pdb=str(pdb), target_description="test", target=[("A", 111), ("A", 114)],
        n_replicas=2, base_seed=1,
        md=MDSettings(equilibration_ps=1.0, save_interval_ps=0.5, platform="CPU"),
        loop=LoopSettings(first_ns=0.002, extend_ns=0.002, max_ns=0.002, no_flip_patience_ns=1.0),
    )
    state.save(tmp_path)
    logs = []
    run_loop(tmp_path, state, SystemFiles(built.topology, built.positions, built.system), logs.append)
    assert state.rounds_ns == [0.002]  # The budget allowed one round only.
    assert (tmp_path / "status.json").exists() and (tmp_path / "decisions.log").exists()

    # A larger budget: resume runs one more round from the saved state.
    state = RunState.load(tmp_path)
    state.loop.max_ns = 0.004
    run_loop(tmp_path, state, log=logs.append)
    assert RunState.load(tmp_path).rounds_ns == [0.002, 0.002]
    trajs = load_run(tmp_path, state.target, 2)
    assert [t.n_frames for t in trajs] == [8, 8]  # 2 parts x 2 ps / 0.5 ps per frame.

    # Pretend round 2 stopped after replica 0 finished it.
    for suffix in (".state.xml", ".dcd", ".csv"):
        (tmp_path / f"replica_1_part001{suffix}").unlink()
    state = RunState.load(tmp_path)
    state.rounds_ns.pop()
    state.pending_round_ns = 0.002
    state.save(tmp_path)

    logs.clear()
    run_loop(tmp_path, RunState.load(tmp_path), log=logs.append)
    assert any("replica 0: round 2 is already complete" in line for line in logs)
    assert RunState.load(tmp_path).rounds_ns == [0.002, 0.002]
    assert [t.n_frames for t in load_run(tmp_path, state.target, 2)] == [8, 8]
