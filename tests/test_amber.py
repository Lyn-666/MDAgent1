"""Prepared Amber systems (--prmtop/--inpcrd) and the restrained equilibration."""

from pathlib import Path

import pytest

from mdagent.build import load_amber, protein_atom_indices
from mdagent.simulate import backbone_atom_indices
from mdagent.target import pdb_summary

DATA = Path(__file__).resolve().parent / "data"
PRMTOP, INPCRD = DATA / "fragment.parm7", DATA / "fragment.rst7"


def test_load_amber_keeps_everything():
    b = load_amber(PRMTOP, INPCRD)
    assert b.system.getNumParticles() == 1343
    assert "Non-protein residues: CL x1, NA x2." in b.notes
    assert len(protein_atom_indices(b.topology)) == 92


def test_info_numbers_amber_residues_like_run():
    assert "Chain 1: protein residues 1-7 (7 residues)" in pdb_summary(PRMTOP)


def test_restraints_are_on_backbone_only():
    b = load_amber(PRMTOP, INPCRD)
    idx = backbone_atom_indices(b.topology)
    atoms = list(b.topology.atoms())
    assert len(idx) == 7 * 3
    assert {atoms[i].name for i in idx} == {"N", "CA", "C"}


def test_box_is_required(tmp_path):
    lines = INPCRD.read_text().splitlines()
    (tmp_path / "nobox.rst7").write_text("\n".join(lines[:-1]) + "\n")  # The last line holds the box.
    with pytest.raises(ValueError, match="no periodic box"):
        load_amber(PRMTOP, tmp_path / "nobox.rst7")


@pytest.mark.slow
def test_run_with_prepared_amber_system(tmp_path):
    import json

    from mdagent.cli import main

    out = tmp_path / "run"
    rc = main(["run", "--prmtop", str(PRMTOP), "--inpcrd", str(INPCRD), "--residues", "3-5",
               "--replicas", "1", "--first-ns", "0.002", "--max-ns", "0.002", "--equil-ps", "3",
               "--save-ps", "0.5", "--platform", "CPU", "--no-enhanced-sampling", "--out", str(out)])
    assert rc == 0
    state = json.loads((out / "run.json").read_text())
    assert state["setup"] == {"amber": [str(PRMTOP), str(INPCRD)]}
    assert state["target"] == [["1", 3], ["1", 4], ["1", 5]]
    # Production starts from a new simulation without the restraint force.
    assert "k_restraint" not in (out / "replica_0_part000.state.xml").read_text()
