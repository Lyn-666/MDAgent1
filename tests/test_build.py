from pathlib import Path

import pytest

from mdagent.build import SetupOptions, build_system
from mdagent.protonation import parse_overrides

DATA = Path(__file__).resolve().parent / "data"
FRAGMENT = DATA / "fragment_108-114.pdb"
FAST = dict(forcefield="amber14", padding_nm=0.8)  # Amber builds fastest on CPU.


@pytest.mark.parametrize("bad", [
    dict(forcefield="gromos"), dict(cation="Mg2+"), dict(box_shape="sphere"),
    dict(ion_conc_molar=-0.1), dict(padding_nm=0.5), dict(ph=15),
])
def test_validate_rejects_bad_options(bad):
    with pytest.raises(ValueError):
        SetupOptions(**bad).validate()


def test_parse_overrides():
    assert parse_overrides("A:31=hip, 70=ASH") == {("A", 31): "HIP", (None, 70): "ASH"}
    with pytest.raises(ValueError):
        parse_overrides("A31HIP")


def test_ions_box_and_report():
    b = build_system(FRAGMENT, SetupOptions(cation="K+", ion_conc_molar=0.3, box_shape="dodecahedron", **FAST))
    names = {r.name for r in b.topology.residues()}
    assert "K" in names and "NA" not in names
    text = b.report()
    assert "dodecahedron" in text and "KCl to 0.3 M" in text
    assert "No missing residues inside the chains." in text
    assert not b.warnings


def test_manual_protonation_choice():
    pdb = Path(__file__).resolve().parents[1] / "examples" / "181L.pdb"
    b = build_system(pdb, SetupOptions(protonation={"A:70": "ASH"}, **FAST))
    asp = next(r for r in b.topology.residues() if r.id == "70")
    assert asp.name == "ASP" and any(a.name == "HD2" for a in asp.atoms())  # Neutral Asp.
    assert "set to ASH" in b.report()


def test_manual_protonation_rejects_wrong_residue_and_chain_end():
    with pytest.raises(ValueError, match="VAL"):
        build_system(FRAGMENT, SetupOptions(protonation={"111": "HIP"}, **FAST))
    with pytest.raises(ValueError, match="chain end"):
        build_system(FRAGMENT, SetupOptions(protonation={"A:108": "GLH"}, **FAST))


def test_internal_gap_from_seqres_is_modeled_and_flagged_near_target():
    b = build_system(DATA / "fragment_gap_seqres.pdb", SetupOptions(**FAST), target=[("A", 109)])
    assert "Missing chain A residues 111-111 (1 residues, inside the chain): modeled" in b.report()
    assert any("near the target region" in w for w in b.warnings)
    assert any(r.id == "111" and r.name == "VAL" for r in b.topology.residues())


def test_gap_without_seqres_is_reported_as_chain_break():
    b = build_system(DATA / "fragment_gap_noseqres.pdb", SetupOptions(**FAST))
    assert any("broken between GLY110 and ALA112" in w for w in b.warnings)


@pytest.mark.slow
def test_propka_marks_his31_charged_in_t4_lysozyme():
    pdb = Path(__file__).resolve().parents[1] / "examples" / "181L.pdb"
    b = build_system(pdb, SetupOptions(check_protonation=True, **FAST))
    his = next(r for r in b.topology.residues() if r.name == "HIS")
    assert {"HD1", "HE2"} <= {a.name for a in his.atoms()}  # HIP: protons on both nitrogens.
    assert "HIS31 chain A: predicted pKa" in b.report()


def test_small_solute_gets_more_padding_for_charmm_cutoff():
    from mdagent.build import _min_box_width_nm
    b = build_system(FRAGMENT, SetupOptions(forcefield="charmm36m", padding_nm=0.8))
    assert _min_box_width_nm(b.topology.getPeriodicBoxVectors()) >= 1.1 * 2 * 1.2
    assert "Increased the padding" in b.report()


def test_dodecahedron_width_is_checked():
    from mdagent.build import _min_box_width_nm
    b = build_system(FRAGMENT, SetupOptions(box_shape="dodecahedron", **FAST))
    assert _min_box_width_nm(b.topology.getPeriodicBoxVectors()) >= 1.1 * 2 * 1.0


@pytest.mark.parametrize("ff", ["amber14", "charmm36m"])
def test_terminal_caps_build_with_both_force_fields(ff):
    capped = build_system(FRAGMENT, SetupOptions(forcefield=ff, padding_nm=0.8, termini="capped"))
    charged = build_system(FRAGMENT, SetupOptions(forcefield=ff, padding_nm=0.8, termini="charged"))
    first = lambda b: next(r for r in b.topology.residues() if r.name not in ("HOH", "NA", "CL"))
    if ff.startswith("amber"):
        assert first(capped).name == "ACE"                     # Amber: caps are residues of their own.
    else:
        names = {a.name for a in first(capped).atoms()}        # CHARMM: caps are part of the end residue.
        assert first(capped).name == "GLU" and {"CAY", "CY", "OY"} <= names
    assert not any(a.name == "OXT" for a in capped.topology.atoms())
    assert any(a.name == "OXT" for a in charged.topology.atoms())
    assert "Chain ends capped" in capped.report() and "charged" in charged.report()


def test_termini_auto_caps_only_ends_that_continue():
    pdb = Path(__file__).resolve().parents[1] / "examples" / "181L.pdb"
    b = build_system(pdb, SetupOptions(**FAST), cap_ends={"A": (False, True)})
    res = [r for r in b.topology.residues() if r.name not in ("HOH", "NA", "CL")]
    assert res[0].name == "MET" and res[-1].name == "NME"
