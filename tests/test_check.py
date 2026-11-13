"""Structure check: reference sequence, tags, chains, metals, gaps. Offline: UniProt files come from tests/data."""

import shutil
from pathlib import Path

import pytest

from mdagent.check import check_structure

DATA = Path(__file__).resolve().parent / "data"
REPO = DATA.parents[1]


@pytest.fixture
def root(tmp_path):
    """A project folder with the UniProt sequences already downloaded."""
    (tmp_path / "structures" / "uniprot").mkdir(parents=True)
    for f in (DATA / "uniprot").glob("*.fasta"):
        shutil.copy(f, tmp_path / "structures" / "uniprot" / f.name)
    return tmp_path


def _texts(result, level):
    return [f.text for f in result.findings if f.level == level]


def test_181l_has_differences_without_explanation(root):
    r = check_structure(REPO / "examples" / "181L.pdb", root=root)
    assert not r.clean
    assert any("R12G" in t and "I137R" in t for t in _texts(r, "problem"))
    assert any("C54T (conflict)" in t and "L99A (conflict)" in t for t in _texts(r, "note"))


def test_4hjo_is_clean_with_explained_mutation_and_unresolved_tag(root):
    r = check_structure(DATA / "4HJO.pdb", root=root)
    assert r.clean, r.report()
    notes = " ".join(_texts(r, "note"))
    assert "V924R (engineered mutation)" in notes
    assert "10 tag or extra residue(s)" in notes and "662-671" in notes
    assert r.remove_residues == []          # The tag has no coordinates, so there is nothing to remove.


def test_4obe_copies_isoform_and_tag_removal(root):
    r = check_structure(DATA / "4OBE.pdb", root=root)
    assert any("copies of the same protein" in t for t in _texts(r, "problem"))
    a = check_structure(DATA / "4OBE.pdb", chains=["A"], root=root)
    assert a.clean, a.report()
    assert any("P01116-2 (an isoform)" in t for t in _texts(a, "note"))
    b = check_structure(DATA / "4OBE.pdb", chains=["B"], root=root)
    assert b.remove_residues == [("B", 0, "")]       # Gly0, an expression tag with coordinates.
    kept = check_structure(DATA / "4OBE.pdb", chains=["B"], keep_tags=True, root=root)
    assert kept.remove_residues == []


def test_wrong_numbering_in_the_file_is_aligned_by_sequence(root, tmp_path):
    text = (DATA / "4HJO.pdb").read_text().replace("EGFR_HUMAN     696   1022", "EGFR_HUMAN     700   1026")
    path = tmp_path / "shifted.pdb"
    path.write_text(text)
    r = check_structure(path, root=root)
    assert r.clean, r.report()
    assert any("aligned to the reference sequence" in t for t in _texts(r, "note"))


def test_structural_metal_is_a_problem(root, tmp_path):
    lines = (DATA / "4HJO.pdb").read_text().splitlines()
    zn = "HETATM 9999 ZN    ZN A1101      10.000  10.000  10.000  1.00 20.00          ZN"
    i = max(n for n, l in enumerate(lines) if l.startswith(("ATOM", "HETATM")))
    path = tmp_path / "zn.pdb"
    path.write_text("\n".join(lines[:i + 1] + [zn] + lines[i + 1:]) + "\n")
    assert any("Metal ion(s) ZN" in t for t in _texts(check_structure(path, root=root), "problem"))


def test_file_without_reference_needs_uniprot(root):
    frag = DATA / "fragment_108-114.pdb"
    r = check_structure(frag, root=root)
    assert any("no reference sequence" in t for t in _texts(r, "problem"))
    r = check_structure(frag, uniprot="P00720", root=root)
    assert r.clean, r.report()


def test_unexplained_chain_break_is_a_problem(root, tmp_path):
    lines = [l for l in (DATA / "4HJO.pdb").read_text().splitlines()
             if not (l.startswith("ATOM") and 750 <= int(l[22:26]) <= 757)]
    path = tmp_path / "break.pdb"
    path.write_text("\n".join(lines) + "\n")
    assert any("broken between residues 749 and 758" in t for t in _texts(check_structure(path, root=root), "problem"))


@pytest.mark.slow
def test_run_stops_on_problems_and_goes_on_after_the_fix(root, monkeypatch):
    import json

    from mdagent.cli import main

    monkeypatch.chdir(root)
    shutil.copy(DATA / "4OBE.pdb", root / "4OBE.pdb")
    base = ["run", "--pdb", "4OBE.pdb", "--around-ligand", "GDP", "--forcefield", "amber14", "--platform", "CPU",
            "--setup-only", "--out", "runs/kras"]
    assert main(base) == 4
    assert (root / "runs" / "kras" / "structure_check.txt").exists()
    assert not (root / "runs" / "kras" / "run.json").exists()
    assert main(base + ["--chains", "A"]) == 0
    state = json.loads((root / "runs" / "kras" / "run.json").read_text())
    assert {c for c, n in state["target"]} == {"A"}
    assert state["setup"]["chains"] == ["A"]


def test_report_uses_project_relative_path_and_explains_caps(root, tmp_path):
    lines = (DATA / "4HJO.pdb").read_text().splitlines()
    cap = "HETATM 9998  N   NH2 A 999      10.000  10.000  10.000  1.00 20.00           N"
    i = max(n for n, l in enumerate(lines) if l.startswith(("ATOM", "HETATM")))
    (root / "structures" / "capped.pdb").write_text("\n".join(lines[:i + 1] + [cap] + lines[i + 1:]) + "\n")
    r = check_structure(root / "structures" / "capped.pdb", root=root)
    assert r.report().startswith("Structure check of structures/capped.pdb:")
    assert any("Terminal cap(s) NH2" in t for t in _texts(r, "note"))
    assert not any("NH2" in t for t in _texts(r, "note") if t.startswith("Other molecule"))


def test_cap_ends_follow_the_reference_sequence(root):
    assert check_structure(DATA / "4HJO.pdb", root=root).cap_ends == {"A": (True, True)}       # Kinase domain only.
    assert check_structure(DATA / "4OBE.pdb", chains=["A"], root=root).cap_ends == {"A": (False, True)}
    r = check_structure(REPO / "examples" / "181L.pdb", root=root)
    assert r.cap_ends == {"A": (False, True)}   # Starts at Met1; residues 163-164 are missing.
    assert any("covers residues 1-162 of the 164-residue reference" in t for t in _texts(r, "note"))
