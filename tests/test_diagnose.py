import numpy as np

from mdagent.diagnose import (
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


def _atom(i, name, resname, resseq, xyz):
    x, y, z = xyz
    return (f"ATOM  {i:5d}  {name:<3s} {resname} A{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}")


def test_proline_and_disulfide_cysteine_are_fixed(tmp_path):
    from mdagent.diagnose import fixed_chi1_residues

    lines = [_atom(1, "SG", "CYS", 3, (0, 0, 0)), _atom(2, "SG", "CYS", 20, (2.04, 0, 0)),  # S-S bond: 2.04 Å
             _atom(3, "SG", "CYS", 30, (9, 0, 0)), _atom(4, "CG", "PRO", 7, (0, 9, 0))]   # Free Cys; Pro.
    pdb = tmp_path / "p.pdb"
    pdb.write_text("\n".join(lines) + "\nEND\n")
    fixed = fixed_chi1_residues(pdb)
    assert set(fixed) == {("A", 3), ("A", 20), ("A", 7)}
    assert "CYS20" in fixed[("A", 3)] and "Proline" in fixed[("A", 7)]


def test_fixed_residues_are_not_judged_and_get_no_enhanced_sampling():
    from pathlib import Path

    import mdtraj as md

    from mdagent.agent import LoopSettings, decide
    from mdagent.diagnose import diagnose_trajectories

    t = md.load(str(Path(__file__).resolve().parents[1] / "examples" / "181L.pdb"))
    target = [("A", 37), ("A", 111), ("A", 114)]           # PRO37, VAL111, PHE114 (pretend it is fixed).
    reports = diagnose_trajectories([t, t], target, 20.0, fixed={("A", 114): "held"})
    by_res = {r.resseq: r for r in reports}
    assert by_res[37].verdict == "fixed_chi1" and "Proline" in by_res[37].reason   # Found by name alone.
    assert by_res[114].verdict == "fixed_chi1" and by_res[114].reason == "held"
    assert by_res[111].verdict == "no_flip"
    decisions, _ = decide(reports, 20.0, LoopSettings(no_flip_patience_ns=10.0))
    assert {d.residue: d.action for d in decisions} == {"VAL111": "enhanced_sampling", "PRO37": "skip",
                                                       "PHE114": "skip"}
