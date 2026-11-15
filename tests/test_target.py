from pathlib import Path

import pytest

from mdagent.target import Target, parse_residue_spec, resolve_target

PDB = Path(__file__).resolve().parents[1] / "examples" / "181L.pdb"


def test_parse_residue_spec():
    assert parse_residue_spec("A:99-101, 111") == (("A", 99), ("A", 100), ("A", 101), (None, 111))


@pytest.mark.parametrize("bad", ["A:120-99", "A99", ""])
def test_parse_residue_spec_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_residue_spec(bad)


def test_pocket_around_benzene_contains_val111():
    keys = resolve_target(Target(ligand="BNZ", cutoff_nm=0.5), PDB)
    assert ("A", 111) in keys
    assert ("A", 99) in keys  # Ala99 is the L99A mutation site lining the cavity


def test_explicit_residues_without_chain():
    assert resolve_target(Target(residues=parse_residue_spec("111")), PDB) == [("A", 111)]


def test_unknown_ligand_lists_what_exists():
    with pytest.raises(ValueError, match="BNZ"):
        resolve_target(Target(ligand="XYZ"), PDB)
