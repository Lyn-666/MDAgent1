"""The MDAgent chat window: a local web page.

    python -m mdagent chat            # then open http://localhost:8765

The language model (Claude Code, Codex or the Anthropic API, see backends.py)
only reads the conversation and proposes one action: answer, start a new run,
stop a run, or resume a run. A proposed start, stop or resume appears as a card
with a button; nothing happens until the user clicks it.

Each conversation has its own history and its own list of runs, saved in
runs/.chats/<id>.json, so different cases stay apart.

Runs are separate processes in their own process group. Their output goes to
runs/<name>.log, and their process ID to runs/<name>.pid. Closing the page or
stopping the chat server does not stop them. A run ends when it finishes, or
when the user stops it from the chat window.

The chat window does not analyze simulation results (project rule).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import backends
from .llm import FLAGS, SWITCHES, RunPlan, to_argv

INPUT_SUFFIXES = (".pdb", ".parm7", ".prmtop", ".rst7", ".inpcrd")
SKIP_DIRS = {"runs", "tests", ".git", ".venv", "node_modules", "mdagent.egg-info", "__pycache__"}
HISTORY_TURNS = 12


class ChatAction(BaseModel):
    """What the assistant proposes in reply to the user's last message."""

    action: Literal["answer", "plan_run", "stop", "resume"] = Field(
        description="answer: only reply (questions, progress, explanations). plan_run: propose a new run. "
                    "stop: propose to stop a running run. resume: propose to continue a stopped run.")
    reply: str = Field(description="Message to the user, in the user's language.")
    run_name: Optional[str] = Field(
        None, description="plan_run: a new short name (lowercase letters, digits, - or _). "
                          "stop/resume: the name of an existing run from the run list.")
    pdb: Optional[str] = Field(None, description="plan_run: PDB input file, from the list of input files.")
    prmtop: Optional[str] = Field(None, description="plan_run: Amber topology file, for a prepared system.")
    inpcrd: Optional[str] = Field(None, description="plan_run: Amber coordinate file, together with prmtop.")
    plan: Optional[RunPlan] = Field(None, description="plan_run: the run settings.")
    fetch_pdb_ids: list[str] = Field(
        default_factory=list,
        description="PDB IDs (for example 3HTB) that the user wants to use but that are not in the list of input "
                    "files yet. MDAgent downloads them to structures/<ID>.pdb and asks you again with the new "
                    "files listed. Empty in all other cases.")


SYSTEM = """You are the chat assistant of MDAgent, a molecular dynamics tool. MDAgent builds a solvated
system (from a protein PDB file, or uses a prepared Amber system that keeps its ligands), runs plain MD in
rounds, finds side chains in a target region that did not sample their chi1 states, and runs enhanced
sampling on them. Runs go on in the background until they finish or the user stops them.

Choose one action for the user's last message:
- plan_run: the user wants a new run. Pick the input file only from the list of input files. Fill in only
  the settings the user asked for and leave the others null (defaults). If the user says "development test"
  or "short", use replicas 2, first_ns 0.1, extend_ns 0.1, max_ns 0.3, no_flip_patience_ns 0.25,
  enhanced_ns 0.1. Never guess a ligand name or residue number: use only names and numbers in the file
  summaries. If anything needed is unclear, use action "answer" and ask.
- stop: the user asks to stop a run that is running. run_name must be in the run list.
- resume: the user asks to continue a run that is stopped or failed. run_name must be in the run list.
- answer: everything else, for example progress questions. Answer progress questions only from the run
  list below.

Before a run that uses a PDB file, MDAgent checks the structure: it compares each chain with its UniProt
reference sequence, finds tags, copies of the same chain, metal ions, unknown residues and long gaps. If the
check finds problems, no button is shown and the user sees the report. Then propose a new plan with the options
the user chooses (chains, keep_tags, uniprot). Set accept_problems to true only if the user explicitly says to go
on despite the problems.

If the user gives a PDB ID (four characters, for example 3HTB) that is not in the list of input files, put it
in fetch_pdb_ids and use action "answer". MDAgent downloads it to structures/<ID>.pdb and asks you again;
then the file and its summary are in the list, and you can plan the run.

Nothing starts or stops until the user clicks the button under your message, so say what the button will
do. Do not analyze simulation results (populations, convergence): the user does that. Write `reply` in the
language of the user's last message (Chinese or English), in short plain sentences."""


# ---------------------------------------------------------------- messages in two languages

TEXT = {
    "zh": {
        "backend_failed": "语言模型后台出错：{e}",
        "cannot_use": "\n\n（MDAgent 无法执行这个提议：{e}）",
        "no_settings": "没有给出运行设置",
        "file_missing": "找不到输入文件 {f}",
        "invalid": "设置不是有效的选项",
        "card_start": "开始新的运行“{name}”",
        "card_stop": "停止运行“{name}”",
        "card_resume": "从最后保存的位置继续运行“{name}”",
        "no_run": "没有名为“{name}”的运行",
        "not_running": "运行“{name}”没有在运行",
        "still_running": "运行“{name}”还在运行",
        "expired": "这个提议已经失效，请重新提出要求。",
        "started": "运行“{name}”已在后台开始（进程 {pid}），日志在 runs/{name}.log。关掉网页也不会停。",
        "resumed": "运行“{name}”已在后台继续（进程 {pid}）。",
        "stopped": "运行“{name}”已停止。",
        "was_not_running": "运行“{name}”没有在运行。",
        "downloaded": "已从 PDB 下载 {id}，保存在 {path}。",
        "already_there": "{id} 之前已经下载过，在 {path}。",
        "download_failed": "下载 {id} 失败：{e}",
        "check_problems": "\n\n结构检查发现了问题，所以还没有生成运行卡片：\n\n{report}\n\n请告诉我怎么处理，"
                          "例如“只保留 A 链”“保留标签”，或者明确说“有问题也继续”。",
    },
    "en": {
        "backend_failed": "The language model backend failed: {e}",
        "cannot_use": "\n\n(MDAgent cannot carry out this proposal: {e})",
        "no_settings": "no run settings were given",
        "file_missing": "input file {f} not found",
        "invalid": "the settings are not valid options",
        "card_start": "Start a new run '{name}'",
        "card_stop": "Stop run '{name}'",
        "card_resume": "Continue run '{name}' from the last saved point",
        "no_run": "there is no run named '{name}'",
        "not_running": "run '{name}' is not running",
        "still_running": "run '{name}' is still running",
        "expired": "This proposal is no longer available. Please ask again.",
        "started": "Run '{name}' started in the background (process {pid}). Log: runs/{name}.log. "
                   "Closing the page does not stop it.",
        "resumed": "Run '{name}' continues in the background (process {pid}).",
        "stopped": "Run '{name}' stopped.",
        "was_not_running": "Run '{name}' was not running.",
        "downloaded": "Downloaded {id} from the PDB to {path}.",
        "already_there": "{id} was downloaded before; it is in {path}.",
        "download_failed": "Download of {id} failed: {e}",
        "check_problems": "\n\nThe structure check found problems, so there is no run button yet:\n\n{report}\n\n"
                          "Tell me what to do, for example \"keep chain A only\" or \"keep the tags\", or say "
                          "clearly that I should go on despite the problems.",
    },
}


def t(lang: str, key: str, **kw) -> str:
    return TEXT.get(lang, TEXT["en"])[key].format(**kw)


# ---------------------------------------------------------------- runs

@dataclass
class Runs:
    root: Path                                  # Project folder; runs live in root / "runs".
    procs: dict = field(default_factory=dict)   # pid -> Popen, for runs started by this server

    @property
    def dir(self) -> Path:
        return self.root / "runs"

    def names(self) -> list[str]:
        if not self.dir.exists():
            return []
        found = {p.name for p in self.dir.iterdir() if (p / "run.json").exists()}
        found |= {p.stem for p in self.dir.glob("*.pid")}
        return sorted(found, key=lambda n: -self._mtime(n))

    def _mtime(self, name: str) -> float:
        files = [self.dir / name, *self._logs(name)]
        return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)

    def _logs(self, name: str) -> list[Path]:
        return sorted(self.dir.glob(f"{name}*.log"), key=lambda p: p.stat().st_mtime)

    def pid(self, name: str) -> int | None:
        f = self.dir / f"{name}.pid"
        return int(f.read_text()) if f.exists() else None

    def alive(self, name: str) -> bool:
        pid = self.pid(name)
        if pid is not None:
            if pid in self.procs:
                return self.procs[pid].poll() is None
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                pass
        # Runs started outside the chat window (for example with nohup) have no pid file.
        out = subprocess.run(["pgrep", "-af", "mdagent"], capture_output=True, text=True).stdout
        target = str(self.dir / name)
        return any(re.search(rf"(--out|--run)\s+\S*{re.escape(target)}(\s|$)", line)
                   or re.search(rf"(--out|--run)\s+runs/{re.escape(name)}(\s|$)", line) for line in out.splitlines())

    def status(self, name: str) -> dict:
        logs = self._logs(name)
        lines = logs[-1].read_text(errors="replace").splitlines()[-300:] if logs else []
        meaningful = [l for l in lines if re.match(
            r"^(Round \d|replica \d+: (minimized|part|\s*\d+%)|enhanced replica|Enhanced sampling|Decision:|"
            r"Loading|Building|Summary|AdaptivePy|error|Traceback|\w*Error)", l)]
        finished = any(l.startswith("Summary") for l in lines)
        failed = any(l.startswith(("Traceback", "error:")) for l in lines)
        running = self.alive(name)
        state = "running" if running else "finished" if finished else "failed" if failed else "stopped"
        info = {"name": name, "state": state, "last": meaningful[-3:], "log": str(logs[-1].relative_to(self.root))
                if logs else None}
        run_json = self.dir / name / "run.json"
        if run_json.exists():
            d = json.loads(run_json.read_text())
            info["plain_md_ns_per_replica"] = round(sum(d.get("rounds_ns", [])), 6)
            info["replicas"] = d.get("n_replicas")
        return info

    def launch(self, name: str, argv: list[str], log_suffix: str = "", program: list[str] | None = None) -> int:
        """Start `python -m mdagent <argv>` (or `program`, in tests) as a detached process."""
        self.dir.mkdir(exist_ok=True)
        log = open(self.dir / f"{name}{log_suffix}.log", "ab")
        program = program or [sys.executable, "-u", "-m", "mdagent"]
        proc = subprocess.Popen(
            [*program, *argv], cwd=self.root, stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,  # Own process group: closing the page or the server does not stop it.
        )
        log.close()
        self.procs[proc.pid] = proc
        (self.dir / f"{name}.pid").write_text(str(proc.pid))
        return proc.pid

    def stop(self, name: str) -> bool:
        pid = self.pid(name)
        if pid is None or not self.alive(name):
            return False
        os.killpg(pid, signal.SIGTERM)
        return True


# ---------------------------------------------------------------- the assistant

def input_files(root: Path, limit: int = 20) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for f in sorted(filenames):
            if f.lower().endswith(INPUT_SUFFIXES):
                out.append(Path(dirpath, f).relative_to(root))
                if len(out) >= limit:
                    return out
    return out


_summary_cache: dict = {}


def file_summary(root: Path, rel: Path) -> str:
    from .target import pdb_summary

    path = root / rel
    if rel.suffix.lower() not in (".pdb", ".parm7", ".prmtop"):
        return ""
    key = (str(path), path.stat().st_mtime)
    if key not in _summary_cache:
        try:
            _summary_cache[key] = pdb_summary(path)
        except Exception as e:  # A broken file should not stop the chat.
            _summary_cache[key] = f"(could not read: {e})"
    return _summary_cache[key]


@dataclass
class Conversation:
    id: str
    title: str = ""
    created: float = 0.0
    updated: float = 0.0
    history: list = field(default_factory=list)   # [{"role": "user"|"assistant", "text": ...}]
    runs: list = field(default_factory=list)      # Names of runs started in this conversation.


class ChatStore:
    """Conversations, one JSON file each in runs/.chats/. They survive a restart of the server."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.chats: dict[str, Conversation] = {}
        if folder.exists():
            for f in folder.glob("*.json"):
                try:
                    c = Conversation(**json.loads(f.read_text()))
                    self.chats[c.id] = c
                except (ValueError, TypeError):
                    pass  # A broken file should not stop the chat.

    def list(self) -> list[Conversation]:
        return sorted(self.chats.values(), key=lambda c: -c.updated)

    def create(self) -> Conversation:
        now = time.time()
        c = Conversation(id=uuid.uuid4().hex[:12], created=now, updated=now)
        self.chats[c.id] = c
        return c

    def get(self, chat_id: str | None) -> Conversation:
        if chat_id and chat_id in self.chats:
            return self.chats[chat_id]
        return self.create()

    def save(self, c: Conversation) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        c.updated = time.time()
        tmp = self.folder / f"{c.id}.tmp"
        tmp.write_text(json.dumps(c.__dict__, ensure_ascii=False, indent=1))
        tmp.replace(self.folder / f"{c.id}.json")


def build_prompt(root: Path, runs: Runs, conv: Conversation, message: str) -> str:
    parts = ["Input files (paths relative to the project folder):"]
    for rel in input_files(root):
        summary = file_summary(root, rel)
        parts.append(f"- {rel}" + (f"\n  " + summary.replace("\n", "\n  ") if summary else ""))
    names = runs.names()
    mine = [n for n in names if n in conv.runs]
    parts.append("\nRuns started in this conversation (newest first):")
    parts += [f"- {json.dumps(runs.status(n), ensure_ascii=False)}" for n in mine] or ["- none"]
    others = [n for n in names if n not in conv.runs][:20]
    if others:
        parts.append("\nOther runs in this project (from other conversations or the command line); use them only "
                     "if the user names one:")
        parts += [f"- {n}: {runs.status(n)['state']}" for n in others]
    parts.append("\nRecent conversation:")
    parts += [f"{h['role']}: {h['text']}" for h in conv.history[-HISTORY_TURNS:]] or ["(none)"]
    parts.append(f"\nUser's last message:\n{message}")
    return "\n".join(parts)


def _clean_name(name: str | None, existing: list[str]) -> str:
    base = re.sub(r"[^a-z0-9_-]+", "-", (name or "run").lower()).strip("-")[:40] or "run"
    candidate, i = base, 2
    while candidate in existing:
        candidate, i = f"{base}-{i}", i + 1
    return candidate


class CheckProblems(Exception):
    """The structure check found problems, and the user has not accepted them."""

    def __init__(self, report: str):
        super().__init__(report)
        self.report = report


@dataclass
class Assistant:
    root: Path
    backend: str
    runs: Runs
    store: ChatStore | None = None
    pending: dict = field(default_factory=dict)   # card id -> card
    call: object = backends.call                  # Replaced in tests.
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        if self.store is None:
            self.store = ChatStore(self.runs.dir / ".chats")

    def message(self, text: str, lang: str = "zh", chat_id: str | None = None) -> dict:
        with self.lock:
            conv = self.store.get(chat_id)
            prompt = build_prompt(self.root, self.runs, conv, text)
            conv.history.append({"role": "user", "text": text})
            if not conv.title:
                conv.title = text.strip().splitlines()[0][:40]
            self.store.save(conv)
        try:
            action = self.call(self.backend, SYSTEM, prompt, ChatAction)
            notes = self._fetch(action.fetch_pdb_ids, lang)
            if notes:  # Ask again, now with the downloaded files in the list.
                prompt = build_prompt(self.root, self.runs, conv, text) + "\n\nMDAgent note: " + " ".join(notes)
                action = self.call(self.backend, SYSTEM, prompt, ChatAction)
            reply, card = action.reply, None
            if notes:
                reply = "\n".join(notes) + "\n\n" + reply
            try:
                card = self._card(action, lang)
            except CheckProblems as e:
                reply += t(lang, "check_problems", report=e.report)
            except (ValueError, SystemExit) as e:
                reply += t(lang, "cannot_use", e=e)
        except Exception as e:
            reply, card = t(lang, "backend_failed", e=e), None
        with self.lock:
            conv.history.append({"role": "assistant", "text": reply})
            self.store.save(conv)
            if card:
                card["chat"] = conv.id
                self.pending[card["id"]] = card
        return {"reply": reply, "card": card, "chat": conv.id, "title": conv.title}

    def _fetch(self, ids: list[str], lang: str) -> list[str]:
        """Download the requested PDB files. Returns one note per ID (empty if nothing was asked)."""
        from .fetch import fetch_pdb

        notes = []
        for pdb_id in ids[:5]:
            try:
                path, new = fetch_pdb(pdb_id, self.root)
                rel = path.relative_to(self.root)
                notes.append(t(lang, "downloaded" if new else "already_there", id=pdb_id.upper(), path=rel))
            except ValueError as e:
                notes.append(t(lang, "download_failed", id=pdb_id, e=e))
        return notes

    def _card(self, a: ChatAction, lang: str = "zh") -> dict | None:
        from .cli import build_parser

        names = self.runs.names()
        if a.action == "plan_run":
            if a.plan is None:
                raise ValueError(t(lang, "no_settings"))
            if a.plan.questions:
                return None
            name = _clean_name(a.run_name, names + [p.name for p in self.runs.dir.glob("*")]
                               if self.runs.dir.exists() else names)
            argv = to_argv(a.plan, a.pdb, f"runs/{name}", prmtop=a.prmtop, inpcrd=a.inpcrd)
            for f in [a.pdb, a.prmtop, a.inpcrd]:
                if f and not (self.root / f).exists():
                    raise ValueError(t(lang, "file_missing", f=f))
            try:
                build_parser().parse_args(argv)  # The same checks as the command line.
            except SystemExit:
                raise ValueError(t(lang, "invalid"))
            check_text = None
            if a.pdb and not a.prmtop:
                from .check import check_structure

                chains = [c.strip() for c in (a.plan.chains or "").split(",") if c.strip()] or None
                check = check_structure(self.root / a.pdb, chains=chains, keep_tags=bool(a.plan.keep_tags),
                                        uniprot=a.plan.uniprot, root=self.root)
                check_text = check.report()
                if not check.clean and not a.plan.accept_problems:
                    raise CheckProblems(check_text)
            changed = [f"{n} = {getattr(a.plan, n)}" for n, _ in FLAGS + SWITCHES
                       if getattr(a.plan, n) not in (None, False)]
            return {"id": uuid.uuid4().hex, "kind": "start", "name": name, "argv": argv,
                    "text": t(lang, "card_start", name=name), "changed": changed, "check": check_text,
                    "command": "python -m mdagent " + " ".join(shlex.quote(x) for x in argv)}
        if a.action in ("stop", "resume"):
            if a.run_name not in names:
                raise ValueError(t(lang, "no_run", name=a.run_name))
            alive = self.runs.alive(a.run_name)
            if a.action == "stop" and not alive:
                raise ValueError(t(lang, "not_running", name=a.run_name))
            if a.action == "resume" and alive:
                raise ValueError(t(lang, "still_running", name=a.run_name))
            text = t(lang, "card_stop" if a.action == "stop" else "card_resume", name=a.run_name)
            return {"id": uuid.uuid4().hex, "kind": a.action, "name": a.run_name, "text": text}
        return None

    def confirm(self, card_id: str, lang: str = "zh") -> str:
        with self.lock:
            card = self.pending.pop(card_id, None)
        if card is None:
            return t(lang, "expired")
        name = card["name"]
        conv = self.store.get(card.get("chat"))
        if card["kind"] == "start":
            pid = self.runs.launch(name, card["argv"])
            conv.runs.append(name)
            msg = t(lang, "started", name=name, pid=pid)
        elif card["kind"] == "resume":
            pid = self.runs.launch(name, ["resume", "--run", f"runs/{name}"], log_suffix=f".resume-{int(time.time())}")
            msg = t(lang, "resumed", name=name, pid=pid)
        else:
            msg = t(lang, "stopped", name=name) if self.runs.stop(name) else t(lang, "was_not_running", name=name)
        with self.lock:
            if name not in conv.runs:
                conv.runs.append(name)  # A run stopped or resumed here now belongs to this conversation too.
            conv.history.append({"role": "assistant", "text": msg})
            self.store.save(conv)
        return msg


# ---------------------------------------------------------------- web server

PAGE = (Path(__file__).parent / "chat_page.html")


def _lang(body: dict) -> str:
    return "en" if body.get("lang") == "en" else "zh"


def make_handler(assistant: Assistant):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # Keep the terminal quiet.
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            from urllib.parse import parse_qs, urlparse

            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/chats":
                self._json([{"id": c.id, "title": c.title, "updated": c.updated, "runs": c.runs}
                            for c in assistant.store.list() if c.history])
            elif url.path == "/api/history":
                c = assistant.store.chats.get(q.get("chat", ""))
                self._json({"history": c.history if c else [], "backend": assistant.backend,
                            "chat": c.id if c else None, "title": c.title if c else ""})
            elif url.path == "/api/runs":
                names = assistant.runs.names()
                if q.get("all") != "1":
                    c = assistant.store.chats.get(q.get("chat", ""))
                    names = [n for n in names if c and n in c.runs]
                self._json([assistant.runs.status(n) for n in names])
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/chat":
                self._json(assistant.message(str(body.get("text", "")), _lang(body), body.get("chat") or None))
            elif self.path == "/api/confirm":
                self._json({"reply": assistant.confirm(str(body.get("id", "")), _lang(body))})
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def serve(root: Path, backend: str, port: int, open_browser: bool = True) -> None:
    backend = backends.choose(backend)
    assistant = Assistant(root=root, backend=backend, runs=Runs(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(assistant))
    url = f"http://localhost:{port}"
    print(f"MDAgent chat window: {url}  (language model backend: {backend})")
    print("Runs keep going when you close the page or stop this server (Ctrl+C).")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChat server stopped. Running simulations go on.")
