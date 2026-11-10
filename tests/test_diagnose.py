import numpy as np

from binding_lang.diagnose import (
    ResidueReport,
    assign_states,
    count_transitions,
    judge,
    occupancy,
)

G_PLUS, T, G_MINUS = 0, 1, 2


def test_assign_states_cores_and_wraparound():
    states = assign_states(np.array([60.0, 179.0, -179.0, -60.0, 110.0]))
    # 110° is outside every core, so it keeps the previous state (g-).
    assert list(states) == [G_PLUS, T, T, G_MINUS, G_MINUS]


def test_noise_near_boundary_is_not_a_flip():
    # Wobble between the g+ core and the gap toward t, never entering the t core.
    angles = np.array([60, 105, 95, 110, 100, 65, 108])
    assert count_transitions(assign_states(angles)) == 0


def test_real_flips_are_counted():
    angles = np.array([60, 60, 120, 180, 180, 120, 60, -60])
    assert count_transitions(assign_states(angles)) == 3


def test_leading_frames_outside_cores_are_ignored():
    states = assign_states(np.array([110.0, 110.0, 180.0, 180.0]))
    assert list(states) == [-1, -1, T, T]
    assert count_transitions(states) == 0
    assert occupancy(states)["t"] == 1.0


def _report(transitions, dominant):
    occ = [{"g+": 0.0, "t": 0.0, "g-": 0.0} for _ in dominant]
    for o, d in zip(occ, dominant):
        o[d] = 1.0
    return ResidueReport("A", 111, "VAL", transitions, occ, dominant)


def test_judge_verdicts():
    r = _report([0, 0, 0], ["t", "t", "t"])
    judge(r)
    assert r.verdict == "no_flip"

    r = _report([1, 0, 2], ["t", "g-", "t"])
    judge(r)
    assert r.verdict == "replicas_disagree"

    r = _report([5, 1, 6], ["t", "t", "t"])
    judge(r)
    assert r.verdict == "rare_flips"

    r = _report([5, 7, 6], ["t", "t", "t"])
    judge(r)
    assert r.verdict == "ok"
