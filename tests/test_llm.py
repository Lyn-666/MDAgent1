"""The ask command. Claude is replaced by fixed plans, so no API key is needed."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from mdagent.cli import build_parser  # noqa: E402
from mdagent.llm import MODEL, RunPlan, ask, make_plan, pdb_summary, to_argv  # noqa: E402

PDB = str(Path(__file__).resolve().parents[1] / "examples" / "181L.pdb")


def _plan(**kw):
    base = dict(target_kind="ligand", ligand="BNZ", questions=[], summary="test")
    base.update(kw)
    return RunPlan(**base)


def test_pdb_summary_lists_chain_and_ligands():
    s = pdb_summary(PDB)
    assert "Chain A: protein residues 1-162" in s
    assert "BNZ x1" in s


def test_to_argv_only_sets_what_the_plan_changes():
    plan = _plan(forcefield="amber19", ion_conc=0.1, cation="K+", check_protonation=True, max_ns=0.3)
    argv = to_argv(plan, "x.pdb", "runs/a")
    assert argv == ["run", "--pdb", "x.pdb", "--out", "runs/a", "--around-ligand", "BNZ",
                    "--forcefield", "amber19", "--ion-conc", "0.1", "--cation", "K+", "--max-ns", "0.3",
                    "--check-protonation"]
    args = build_parser().parse_args(argv)
    assert args.ph == 7.0 and args.replicas == 3  # Untouched options keep their defaults.


def test_to_argv_residue_target():
    argv = to_argv(_plan(target_kind="residues", ligand=None, residues="A:99-120"), "x.pdb", "o")
    assert argv[-2:] == ["--residues", "A:99-120"]
    with pytest.raises(ValueError):
        to_argv(_plan(target_kind="residues", ligand=None), "x.pdb", "o")


def test_questions_stop_the_run(tmp_path, capsys):
    plan = _plan(ligand=None, questions=["Which ligand do you mean? The file has BNZ, HED and CL."])
    rc = ask("look at the ATP pocket", PDB, str(tmp_path / "run"), planner=lambda r, s: plan)
    assert rc == 3
    assert "Which ligand" in capsys.readouterr().out
    assert not (tmp_path / "run").exists()


def test_declining_does_not_start(tmp_path, capsys):
    rc = ask("CHARMM, benzene pocket", PDB, str(tmp_path / "run"),
             planner=lambda r, s: _plan(forcefield="charmm36m"), confirm=lambda prompt: "n")
    out = capsys.readouterr().out
    assert rc == 0 and "Not started." in out
    assert "--around-ligand BNZ --forcefield charmm36m" in out
    assert not (tmp_path / "run").exists()


def test_make_plan_request_shape():
    seen = {}

    class FakeMessages:
        def parse(self, **kw):
            seen.update(kw)
            return SimpleNamespace(stop_reason="end_turn", parsed_output=_plan())

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    plan = make_plan("benzene pocket", "Chain A: ...", client=client)
    assert plan.ligand == "BNZ"
    assert seen["model"] == MODEL and seen["output_format"] is RunPlan
    assert seen["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in seen["betas"]
    assert "Chain A: ..." in seen["system"]


def test_refusal_is_reported():
    class FakeMessages:
        def parse(self, **kw):
            return SimpleNamespace(stop_reason="refusal", parsed_output=None)

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    with pytest.raises(ValueError, match="refusal"):
        make_plan("x", "y", client=client)
