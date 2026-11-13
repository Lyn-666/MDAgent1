"""The chat window. Language model backends are replaced by fixed answers."""

import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

from mdagent import backends  # noqa: E402
from mdagent.chat import Assistant, ChatAction, Runs, make_handler  # noqa: E402
from mdagent.llm import RunPlan  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SLEEPER = [sys.executable, "-c", "import time; print('Round 1: running 2 replicas', flush=True); time.sleep(60)"]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "examples").mkdir()
    shutil.copy(REPO / "examples" / "181L.pdb", tmp_path / "examples" / "181L.pdb")
    return tmp_path


def _plan(**kw):
    base = dict(target_kind="ligand", ligand="BNZ", questions=[], summary="s", forcefield="amber14")
    base.update(kw)
    return RunPlan(**base)


def test_strict_schema_for_codex():
    schema = backends.strict_schema(ChatAction)

    def check(node):
        if isinstance(node, dict):
            assert "default" not in node and "title" not in node
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)
    check(schema)


def test_claude_backend_command_and_answer():
    seen = {}

    def runner(cmd, **kw):
        seen["cmd"], seen["cwd"] = cmd, kw["cwd"]
        out = {"result": "", "structured_output": {"action": "answer", "reply": "hi", "run_name": None,
                                                    "pdb": None, "prmtop": None, "inpcrd": None, "plan": None}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(out), stderr="")

    a = backends.call("claude", "SYS", "USER", ChatAction, runner=runner)
    assert a.reply == "hi"
    cmd = seen["cmd"]
    assert cmd[:3] == ["claude", "-p", "USER"] and "--json-schema" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""          # No tools: it can only answer.
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert seen["cwd"] != str(REPO)                     # Not in the project, so no CLAUDE.md is loaded.


def test_claude_backend_without_structured_output_fails():
    runner = lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=json.dumps({"result": "sorry"}), stderr="")
    with pytest.raises(ValueError, match="no structured answer"):
        backends.call("claude", "S", "U", ChatAction, runner=runner)


def test_codex_backend_reads_output_file():
    def runner(cmd, **kw):
        out = Path(cmd[cmd.index("-o") + 1])
        assert json.loads(Path(cmd[cmd.index("--output-schema") + 1]).read_text())["additionalProperties"] is False
        assert cmd[-1].startswith("SYS")                # Instructions come first in the prompt.
        out.write_text(json.dumps({"action": "answer", "reply": "ok", "run_name": None, "pdb": None,
                                   "prmtop": None, "inpcrd": None, "plan": None}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    assert backends.call("codex", "SYS", "U", ChatAction, runner=runner).reply == "ok"


def test_runs_are_detached_and_can_be_stopped(project):
    runs = Runs(project)
    pid = runs.launch("demo", [], program=SLEEPER)
    try:
        assert os.getpgid(pid) == pid                   # Own process group: survives the chat server.
        time.sleep(1)
        st = runs.status("demo")
        assert st["state"] == "running" and st["last"] == ["Round 1: running 2 replicas"]
        assert runs.stop("demo")
        for _ in range(50):
            if not runs.alive("demo"):
                break
            time.sleep(0.1)
        assert runs.status("demo")["state"] == "stopped"
    finally:
        try:
            os.killpg(pid, 9)
        except OSError:
            pass


def _assistant(project, action):
    return Assistant(root=project, backend="claude", runs=Runs(project),
                     call=lambda backend, system, prompt, model: action)


def test_plan_becomes_a_start_card_and_runs_only_after_confirm(project):
    action = ChatAction(action="plan_run", reply="I will run it.", run_name="Dev Test",
                        pdb="examples/181L.pdb", plan=_plan(max_ns=0.3))
    a = _assistant(project, action)
    started = []
    a.runs.launch = lambda name, argv, **kw: started.append((name, argv)) or 123
    r = a.message("amber, benzene pocket, development test")
    card = r["card"]
    assert card["kind"] == "start" and card["name"] == "dev-test"
    assert card["argv"][:5] == ["run", "--pdb", "examples/181L.pdb", "--out", "runs/dev-test"]
    assert started == []                                 # Nothing runs before the click.
    assert "已在后台开始" in a.confirm(card["id"])
    assert started and started[0][0] == "dev-test"
    assert "已经失效" in a.confirm(card["id"])  # A card works once.


def test_messages_follow_the_page_language(project):
    stop = ChatAction(action="stop", reply="stopping", run_name="ghost")
    assert "there is no run named 'ghost'" in _assistant(project, stop).message("stop ghost", lang="en")["reply"]
    assert "没有名为“ghost”的运行" in _assistant(project, stop).message("停止 ghost", lang="zh")["reply"]


def test_prompt_contains_files_and_runs(project):
    seen = {}

    def call(backend, system, prompt, model):
        seen["prompt"] = prompt
        return ChatAction(action="answer", reply="ok")

    a = Assistant(root=project, backend="claude", runs=Runs(project), call=call)
    a.message("what can I simulate?")
    assert "examples/181L.pdb" in seen["prompt"] and "BNZ x1" in seen["prompt"]


def test_questions_and_bad_proposals_give_no_card(project):
    q = ChatAction(action="plan_run", reply="Which ligand?", run_name="x", pdb="examples/181L.pdb",
                   plan=_plan(questions=["Which ligand?"]))
    assert _assistant(project, q).message("atp pocket")["card"] is None
    missing = ChatAction(action="plan_run", reply="ok", run_name="x", pdb="nope.pdb", plan=_plan())
    r = _assistant(project, missing).message("run nope.pdb")
    assert r["card"] is None and "找不到输入文件" in r["reply"]
    stop = ChatAction(action="stop", reply="stopping", run_name="ghost")
    assert "没有名为" in _assistant(project, stop).message("stop ghost")["reply"]


def test_web_server(project):
    a = _assistant(project, ChatAction(action="answer", reply="你好"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(a))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"MDAgent" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/api/runs").read()) == []
        req = urllib.request.Request(base + "/api/chat", data=json.dumps({"text": "hi"}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["reply"] == "你好"
        hist = json.loads(urllib.request.urlopen(base + "/api/history").read())
        assert [m["role"] for m in hist["history"]] == ["user", "assistant"]
    finally:
        server.shutdown()
