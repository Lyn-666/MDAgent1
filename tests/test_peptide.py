"""Peptides built from a sequence (peptide.py) and the secondary-structure quantity (observables.diagnose_secondary)."""

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

from mdagent.peptide import Peptide, parse_sequence, write_pdb  # noqa: E402


def test_parse_sequence_reads_end_groups():
    p = parse_sequence("Ac-AAKAA-NH2")
    assert p.residues == ["ALA", "ALA", "LYS", "ALA", "ALA"] and p.n_cap and p.c_cap and p.text == "Ac-AAKAA-NH2"
    p = parse_sequence(" gsw " .upper())
    assert not p.n_cap and not p.c_cap and p.one_letter == "GSW"
    p = parse_sequence("H-GSWTW-OH")
    assert not p.n_cap and not p.c_cap and p.text == "GSWTW"


@pytest.mark.parametrize("bad, words", [("Ac-AAkA-NH2", "D-amino"), ("AAXB", "standard amino acids"),
                                        ("", "empty"), ("A", "at least 2")])
def test_parse_sequence_refuses_what_it_cannot_build(bad, words):
    with pytest.raises(ValueError, match=words):
        parse_sequence(bad)


@pytest.mark.parametrize("conformation, phi, psi", [("extended", -135, 135), ("helix", -57, -47)])
def test_backbone_has_the_asked_dihedrals_and_bond_lengths(tmp_path, conformation, phi, psi):
    path = write_pdb(parse_sequence("Ac-AAKAAPAKAA-NH2"), tmp_path / "p.pdb", conformation)
    t = md.load(str(path))
    phis = np.degrees(md.compute_phi(t)[1][0])
    psis = np.degrees(md.compute_psi(t)[1][0])
    pro = [i for i, r in enumerate(t.topology.residues) if r.name == "PRO"][0]
    assert np.allclose(np.delete(phis, pro - 1), phi, atol=0.5) and abs(phis[pro - 1] + 65) < 0.5
    assert np.allclose(psis, psi, atol=0.5)
    ca = t.xyz[0][t.topology.select("name CA")]
    assert np.allclose(np.linalg.norm(ca[1:] - ca[:-1], axis=1), 0.380, atol=0.01)


def _frames(tmp_path, n_helix, n_extended):
    """A backbone-only trajectory: n_helix helical frames, then n_extended extended frames."""
    seq = parse_sequence("AAAAKAAAAKAA")
    traj = []
    for conf, n in (("helix", n_helix), ("extended", n_extended)):
        t = md.load(str(write_pdb(seq, tmp_path / f"{conf}.pdb", conf)))
        traj += [t] * n
    return md.join(traj)


def test_secondary_structure_populations_are_right(tmp_path):
    from mdagent.observables import SS_STATES, diagnose_secondary

    trajs = [_frames(tmp_path, 30, 10), _frames(tmp_path, 30, 10)]
    reports = diagnose_secondary(trajs, [("A", 5), ("A", 6)], ns_per_replica=1.0, precision=0.05)
    assert len(reports) == 2
    for r in reports:
        assert r.quantity == "secondary" and set(r.populations) == set(SS_STATES)
        assert r.populations["helix"] == pytest.approx(0.75) and r.populations["coil"] == pytest.approx(0.25)
        assert r.transitions == [1, 1]          # one change from helix to coil in each replica


def test_chat_plan_with_a_sequence_becomes_a_valid_run():
    from mdagent.cli import build_parser
    from mdagent.llm import RunPlan, to_argv

    plan = RunPlan(target_kind="secondary", sequence="Ac-AAKAAAAKAA-NH2", start_conformation="helix",
                   questions=[], summary="x")
    argv = to_argv(plan, None, "runs/p")
    assert argv[argv.index("--sequence") + 1] == "Ac-AAKAAAAKAA-NH2" and "--pdb" not in argv
    args = build_parser().parse_args(argv)
    assert args.quantity == "secondary" and args.start_conformation == "helix"
    with pytest.raises(ValueError, match="standard amino acids"):
        to_argv(RunPlan(target_kind="secondary", sequence="AAZ", questions=[], summary="x"), None, "runs/p")


def test_secondary_target_leaves_out_the_chain_ends(tmp_path):
    from argparse import Namespace

    from mdagent.cli import _interior_residues

    path = write_pdb(parse_sequence("AAKAAA"), tmp_path / "p.pdb")
    target = _interior_residues(Namespace(pdb=str(path)))
    assert [k[1] for k in target.residues] == [2, 3, 4, 5]


def test_peptide_dataclass_one_letter_round_trip():
    assert Peptide(["GLY", "TRP"], False, True, "GW-NH2").one_letter == "GW"
