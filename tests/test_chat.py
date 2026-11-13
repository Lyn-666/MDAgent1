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
    (tmp_path / "structures" / "uniprot").mkdir(parents=True)   # UniProt sequences, so tests run offline.
    for f in (REPO / "tests" / "data" / "uniprot").glob("*.fasta"):
        shutil.copy(f, tmp_path / "structures" / "uniprot" / f.name)
    return tmp_path


def _plan(**kw):
    # 181L differs from its UniProt entry at two positions; most tests accept that and look at other things.
    base = dict(target_kind="ligand", ligand="BNZ", questions=[], summary="s", forcefield="amber14",
                accept_problems=True)
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
                                                    "pdb": None, "prmtop": None, "inpcrd": None, "plan": None, "fetch_pdb_ids": []}}
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
                                   "prmtop": None, "inpcrd": None, "plan": None, "fetch_pdb_ids": []}))
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
    get = lambda path: json.loads(urllib.request.urlopen(base + path).read())

    def post(path, data):
        req = urllib.request.Request(base + path, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    try:
        assert b"MDAgent" in urllib.request.urlopen(base + "/").read()
        assert get("/api/runs") == [] and get("/api/chats") == []
        r = post("/api/chat", {"text": "第一个案例"})
        assert r["reply"] == "你好" and r["title"] == "第一个案例"
        hist = get(f"/api/history?chat={r['chat']}")
        assert [m["role"] for m in hist["history"]] == ["user", "assistant"]
        r2 = post("/api/chat", {"text": "second case"})            # No chat id: a new conversation.
        assert r2["chat"] != r["chat"]
        assert {c["title"] for c in get("/api/chats")} == {"第一个案例", "second case"}
    finally:
        server.shutdown()


def test_conversations_keep_cases_apart(project):
    plan = ChatAction(action="plan_run", reply="ok", run_name="case-a", pdb="examples/181L.pdb", plan=_plan())
    a = _assistant(project, plan)
    a.runs.launch = lambda name, argv, **kw: 1
    r = a.message("case A")
    a.confirm(r["card"]["id"])
    other = a.message("case B")                                # A second conversation.
    assert other["chat"] != r["chat"]
    assert a.store.chats[r["chat"]].runs == ["case-a"]
    assert a.store.chats[other["chat"]].runs == []
    # The prompt of conversation B lists case-a only as an "other run".
    seen = {}
    a.call = lambda backend, system, prompt, model: seen.setdefault("p", prompt) and ChatAction(action="answer", reply="x")
    a.message("how is it going?", chat_id=other["chat"])
    mine = seen["p"].split("Runs started in this conversation")[1].split("Recent conversation")[0]
    assert "- none" in mine.split("Other runs")[0]


def test_conversations_survive_a_restart(project):
    a = _assistant(project, ChatAction(action="answer", reply="hi"))
    r = a.message("remember me")
    b = _assistant(project, ChatAction(action="answer", reply="hi"))  # A new server process.
    c = b.store.chats[r["chat"]]
    assert c.title == "remember me" and [m["text"] for m in c.history] == ["remember me", "hi"]


def test_pdb_id_is_downloaded_then_the_model_is_asked_again(project, monkeypatch):
    import mdagent.fetch as fetch

    class Response:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.data

    pdb_text = (REPO / "examples" / "181L.pdb").read_bytes()
    urls = []
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda url, timeout: urls.append(url) or Response(pdb_text))
    prompts = []

    def call(backend, system, prompt, model):
        prompts.append(prompt)
        if len(prompts) == 1:
            return ChatAction(action="answer", reply="I will download it.", fetch_pdb_ids=["3htb"])
        return ChatAction(action="plan_run", reply="Ready.", run_name="t4", pdb="structures/3HTB.pdb", plan=_plan())

    a = Assistant(root=project, backend="claude", runs=Runs(project), call=call)
    r = a.message("use PDB 3HTB, benzene pocket", lang="en")
    assert urls == ["https://files.rcsb.org/download/3HTB.pdb"]
    assert (project / "structures" / "3HTB.pdb").exists()
    assert "structures/3HTB.pdb" in prompts[1] and "BNZ x1" in prompts[1]   # The second call sees the file.
    assert r["reply"].startswith("Downloaded 3HTB from the PDB to structures/3HTB.pdb.")
    assert r["card"]["argv"][:3] == ["run", "--pdb", "structures/3HTB.pdb"]
    # Asking again does not download again.
    a.call = lambda *x: ChatAction(action="answer", reply="ok", fetch_pdb_ids=["3HTB"])
    a.message("3HTB again", lang="zh")
    assert len(urls) == 1


def test_fetch_rejects_bad_ids_and_reports_missing_entries(tmp_path):
    import urllib.error

    from mdagent.fetch import fetch_pdb

    with pytest.raises(ValueError, match="not a PDB ID"):
        fetch_pdb("abcd", tmp_path)

    def not_found(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    with pytest.raises(ValueError, match="mmCIF"):
        fetch_pdb("9xyz", tmp_path, opener=not_found)
    assert not (tmp_path / "structures" / "9XYZ.pdb").exists()


def test_structure_problems_give_no_card_until_accepted(project):
    plan = ChatAction(action="plan_run", reply="ok", run_name="t4", pdb="examples/181L.pdb",
                      plan=_plan(accept_problems=None))
    r = _assistant(project, plan).message("181L, benzene pocket", lang="en")
    assert r["card"] is None
    assert "structure check found problems" in r["reply"] and "R12G" in r["reply"]
    ok = ChatAction(action="plan_run", reply="ok", run_name="t4", pdb="examples/181L.pdb", plan=_plan())
    r = _assistant(project, ok).message("go on anyway", lang="en")
    assert r["card"] and "--accept-problems" in r["card"]["argv"] and "R12G" in r["card"]["check"]
