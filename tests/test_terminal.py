import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from mdagent import backends  # noqa: E402
from mdagent.chat import Assistant, ChatAction, Runs  # noqa: E402
from mdagent.llm import RunPlan  # noqa: E402
from mdagent.terminal import run_terminal, start  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "examples").mkdir()
    shutil.copy(REPO / "examples" / "181L.pdb", tmp_path / "examples" / "181L.pdb")
    (tmp_path / "structures" / "uniprot").mkdir(parents=True)
    for f in (REPO / "tests" / "data" / "uniprot").glob("*.fasta"):
        shutil.copy(f, tmp_path / "structures" / "uniprot" / f.name)
    return tmp_path


def _scripted(lines):
    it = iter(lines)
    return lambda prompt="": next(it)


def test_terminal_card_runs_only_after_yes(project):
    plan = RunPlan(target_kind="ligand", ligand="BNZ", questions=[], summary="s", forcefield="amber14",
                   accept_problems=True, max_ns=0.3)
    calls = []

    def call(backend, system, prompt, model):
        calls.append(prompt)
        return ChatAction(action="plan_run", reply="I will run it.", run_name="dev", pdb="examples/181L.pdb", plan=plan)

    a = Assistant(root=project, backend="claude", runs=Runs(project), call=call)
    started = []
    a.runs.launch = lambda name, argv, **kw: started.append(name) or 123
    out = []
    # First proposal declined, second accepted; /runs and /help use no language model.
    run_terminal(a, "en", read=_scripted(["/help", "run 181L", "n", "run 181L", "y", "/runs", "/quit"]),
                 out=out.append)
    text = "\n".join(out)
    assert len(calls) == 2 and started == ["dev"]
    assert "Nothing was done." in text and "started in the background" in text
    assert "Command: python -m mdagent run --pdb examples/181L.pdb --out runs/dev" in text
    assert "/runs" in text and "Bye." in text


def test_terminal_conversation_is_shared_with_the_web_page(project):
    a = Assistant(root=project, backend="claude", runs=Runs(project),
                  call=lambda *args: ChatAction(action="answer", reply="你好"))
    run_terminal(a, "zh", read=_scripted(["第一个案例", "/quit"]), out=lambda s: None)
    saved = list((project / "runs" / ".chats").glob("*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text())["title"] == "第一个案例"


def test_start_asks_and_remembers_the_choices(project, monkeypatch):
    monkeypatch.setattr(backends, "available", lambda: ["claude", "codex"])
    out = []
    # English, the second backend (codex), terminal; then leave at once.
    assert start(project, read=_scripted(["2", "2", "2", "/quit"]), out=out.append) == 0
    assert json.loads((project / "runs" / ".chats" / "start.json").read_text()) == \
        {"lang": "en", "backend": "codex", "ui": "terminal"}
    assert any("language model: ChatGPT" in line for line in out)
    out.clear()
    # Next time Enter keeps the saved choices.
    start(project, read=_scripted(["", "", "", "/quit"]), out=out.append)
    assert any("language model: ChatGPT" in line for line in out)


def test_start_without_a_language_model_explains_what_to_do(project, monkeypatch):
    monkeypatch.setattr(backends, "available", lambda: [])
    out = []
    assert start(project, lang="zh", read=_scripted([]), out=out.append) == 2
    assert "Claude Code" in out[0] and "mdagent run --help" in out[0]


def test_ask_uses_the_chosen_login(tmp_path, monkeypatch, capsys):
    from mdagent.llm import ask

    seen = {}

    def call(backend, system, user, model):
        seen["backend"] = backend
        return RunPlan(target_kind="ligand", ligand="ATP", questions=["Which ligand?"], summary="s")

    monkeypatch.setattr(backends, "call", call)
    monkeypatch.setattr(backends, "available", lambda: ["codex"])
    assert ask("atp pocket", str(REPO / "examples" / "181L.pdb"), str(tmp_path / "r")) == 3
    assert seen["backend"] == "codex" and "(codex)" in capsys.readouterr().out


def test_prompt_says_how_the_user_confirms(project):
    seen = []
    call = lambda backend, system, prompt, model: seen.append(prompt) or ChatAction(action="answer", reply="ok")  # noqa: E731
    Assistant(root=project, backend="claude", runs=Runs(project), call=call, ui="terminal").message("hi")
    Assistant(root=project, backend="claude", runs=Runs(project), call=call).message("hi")
    assert seen[0].startswith("Interface: terminal") and seen[1].startswith("Interface: web page")


def test_development_test_can_shorten_equilibration():
    from mdagent.llm import to_argv

    plan = RunPlan(target_kind="ligand", ligand="BNZ", questions=[], summary="s", equil_ps=60)
    assert to_argv(plan, "x.pdb", "runs/a")[-2:] == ["--equil-ps", "60"]
