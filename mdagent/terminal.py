"""MDAgent in the terminal, and the `start` menu.

    python -m mdagent start                     # choose language, language model and web page or terminal
    python -m mdagent chat --terminal           # talk in this terminal

The terminal mode uses the same assistant as the web page (chat.py): the same
steps, the same structure check, and the same confirmation before anything
starts, stops or resumes. Conversations are saved in runs/.chats/, so the web
page shows them too.

Commands that start with "/" are handled by MDAgent itself and do not use the
language model, so they cost no tokens: /runs, /new, /help, /quit.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable

from . import backends

TEXT = {
    "zh": {
        "hello": "MDAgent 终端对话（语言模型：{backend}）。用一句话说要跑什么，按 Enter 发送。输入 /help 查看命令。",
        "help": "命令（不用语言模型，不花 token）：\n"
                "  /runs  列出这个项目的运行：状态、最近的进度和结果文件夹\n"
                "  /last  在可以翻页的窗口里看上一条回复（长回复用它看全；按 q 回来）\n"
                "  /report [运行名]  在翻页窗口里看结果报告（不写运行名就看最近的运行）\n"
                "  /new   开始一个新对话（不同的案例用不同的对话）\n"
                "  /quit  退出（正在跑的模拟不会停）\n"
                "其他输入都发给语言模型，例如：“用 181L，看苯周围的口袋，开发测试，有问题也继续”。",
        "you": "你> ", "agent": "MDAgent> ", "thinking": "（思考中…）",
        "changed": "改动的设置：", "none": "无", "command": "命令：", "check": "结构检查报告：",
        "ask_start": "开始运行？", "ask_stop": "停止运行？", "ask_resume": "继续运行？", "yes_no": " [y/N] ",
        "not_done": "没有执行。", "new_chat": "已开始新对话。", "no_last": "还没有回复可看。", "no_report": "{name} 还没有结果报告（运行结束时才写）。", "bye": "再见。正在跑的模拟会继续在后台运行。",
        "plain_md": "普通 MD 每条 {ns:g} ns", "no_runs": "还没有运行。", "states": {"running": "运行中", "finished": "已完成", "stopped": "已停止",
                                            "failed": "出错"},
        "folder": "结果文件在电脑“{host}”的 {folder}（{n} 个文件）",
    },
    "en": {
        "hello": "MDAgent in the terminal (language model: {backend}). Say in one sentence what to run, then press "
                 "Enter. Type /help for commands.",
        "help": "Commands (no language model, no tokens):\n"
                "  /runs  list the runs of this project: state, latest progress and result folder\n"
                "  /last  read the last reply in a window you can page through (for long replies; q to return)\n"
                "  /report [run]  read the result report in that window (without a name: the latest run)\n"
                "  /new   start a new conversation (use one conversation per case)\n"
                "  /quit  leave (running simulations go on)\n"
                "Everything else goes to the language model, for example: \"use 181L, the pocket around benzene, "
                "development test, go on despite problems\".",
        "you": "you> ", "agent": "MDAgent> ", "thinking": "(thinking…)",
        "changed": "Changed settings: ", "none": "none", "command": "Command: ", "check": "Structure check report:",
        "ask_start": "Start the run?", "ask_stop": "Stop the run?", "ask_resume": "Continue the run?",
        "yes_no": " [y/N] ", "not_done": "Nothing was done.", "new_chat": "Started a new conversation.",
        "no_last": "No reply to show yet.", "no_report": "{name} has no result report yet (it is written when the run ends).",
        "bye": "Bye. Running simulations go on in the background.",
        "plain_md": "plain MD {ns:g} ns per replica", "no_runs": "No runs yet.", "states": {"running": "running", "finished": "finished", "stopped": "stopped",
                                             "failed": "failed"},
        "folder": "Result files on computer '{host}' in {folder} ({n} files)",
    },
}

BACKEND_NAMES = {"claude": "Claude（Claude Code）", "codex": "ChatGPT（Codex）", "api": "Claude API"}


def _enable_line_editing() -> None:
    try:
        import readline  # noqa: F401  (arrow keys and history in input())
    except ImportError:
        pass


def show_runs(assistant, lang: str, out: Callable[[str], None]) -> None:
    T = TEXT[lang]
    names = assistant.runs.names()
    if not names:
        out(T["no_runs"])
        return
    for name in names:
        st = assistant.runs.status(name)
        ns = st.get("plain_md_ns_per_replica")
        head = f"- {name}: {T['states'].get(st['state'], st['state'])}" + (", " + T["plain_md"].format(ns=ns) if ns else "")
        out(head)
        for line in st.get("last", []):
            out(f"    {line}")
        if st.get("files"):
            out("    " + T["folder"].format(host=st["host"], folder=st["folder"], n=len(st["files"])))


def show_card(card: dict, lang: str, out: Callable[[str], None]) -> None:
    T = TEXT[lang]
    out(f"\n[ {card['text']} ]")
    if card.get("save_text"):
        out(card["save_text"])
    if "changed" in card:
        out(T["changed"] + (", ".join(card["changed"]) if card["changed"] else T["none"]))
    if card.get("check"):
        out(T["check"])
        out("  " + card["check"].replace("\n", "\n  "))
    if card.get("command"):
        out(T["command"] + card["command"])


def page(text: str) -> None:
    """Show text in a pager (less), so a long reply can be read in full whatever the terminal does with the
    mouse wheel (tmux without mouse mode turns it into arrow keys, which scroll the input history)."""
    import pydoc

    pydoc.pager(text)


def _watch_run_messages(assistant, state: dict, lang: str, out, stop: threading.Event,
                        every_s: float = 20.0) -> None:
    """Print messages that runs write while the user is not typing (a suggestion, results ready ...).
    Nothing is printed while the pager is open (state["paging"]); the messages come after it."""
    while not stop.wait(every_s):
        if state.get("paging"):
            continue
        for m in assistant.sync(state.get("chat"), lang):
            out("\n" + TEXT[lang]["agent"] + m["text"])
            state["last"] = m["text"]


def run_terminal(assistant, lang: str = "zh", read: Callable[[str], str] = input,
                 out: Callable[[str], None] = print, watch_every_s: float = 20.0,
                 pager: Callable[[str], None] = page) -> int:
    """Talk with the assistant in the terminal until /quit, Ctrl+D or Ctrl+C."""
    T = TEXT[lang]
    if read is input:
        _enable_line_editing()
    out(T["hello"].format(backend=BACKEND_NAMES.get(assistant.backend, assistant.backend)))
    state = {"chat": None, "last": ""}

    def local(text: str) -> bool:
        """Commands that need no language model. Returns True if `text` was one of them."""
        if text == "/help":
            out(T["help"])
        elif text == "/runs":
            show_runs(assistant, lang, out)
        elif text.split()[0] == "/report":
            names = assistant.runs.names()
            parts = text.split()
            name = parts[1] if len(parts) > 1 else (max(names, key=lambda n: (assistant.runs.dir / n).stat().st_mtime)
                                                    if names else "")
            report = assistant.runs.dir / name / f"report.{lang}.md" if name else None
            if not report or not report.exists():
                out(T["no_report"].format(name=name or "-"))
            else:
                state["paging"] = True
                try:
                    pager(report.read_text(encoding="utf-8"))
                finally:
                    state["paging"] = False
        elif text == "/last":
            if not state["last"]:
                out(T["no_last"])
            else:
                state["paging"] = True
                try:
                    pager(T["agent"] + state["last"])
                finally:
                    state["paging"] = False
        else:
            return False
        return True
    stop = threading.Event()
    threading.Thread(target=_watch_run_messages, args=(assistant, state, lang, out, stop, watch_every_s),
                     daemon=True).start()
    try:
        while True:
            for m in assistant.sync(state["chat"], lang):
                out(T["agent"] + m["text"])
                state["last"] = m["text"]
            text = read("\n" + T["you"]).strip()
            if not text:
                continue
            if text in ("/quit", "/exit", "/q"):
                break
            if local(text):
                continue
            if text == "/new":
                state["chat"] = None
                out(T["new_chat"])
                continue
            out(T["thinking"])
            r = assistant.message(text, lang, state["chat"])
            state["chat"] = r["chat"]
            out(T["agent"] + r["reply"])
            state["last"] = r["reply"]
            card = r.get("card")
            if card:
                show_card(card, lang, out)
                while True:   # A command such as /runs or /last is done first; then the question comes again.
                    answer = read(T[f"ask_{card['kind']}"] + T["yes_no"]).strip().lower()
                    if not (answer.startswith("/") and local(answer)):
                        break
                if answer in ("y", "yes", "是", "好", "开始", "继续", "停止"):
                    out(T["agent"] + assistant.confirm(card["id"], lang))
                else:
                    assistant.pending.pop(card["id"], None)
                    out(T["not_done"])
    except (EOFError, KeyboardInterrupt):
        out("")
    stop.set()
    out(T["bye"])
    return 0


# ---------------------------------------------------------------- the start menu

def _choose(question: str, options: list[tuple[str, str]], default: str | None,
            read: Callable[[str], str], out: Callable[[str], None]) -> str:
    """Ask for one of the options (key, label). Enter keeps the default."""
    keys = [k for k, _ in options]
    if default not in keys:
        default = keys[0]
    out(question)
    for i, (k, label) in enumerate(options, 1):
        out(f"  {i}) {label}" + ("   ←" if k == default else ""))
    while True:
        a = read(f"[1-{len(options)}, Enter = {keys.index(default) + 1}] ").strip()
        if not a:
            return default
        if a.isdigit() and 1 <= int(a) <= len(options):
            return keys[int(a) - 1]


def start(root: Path, backend: str | None = None, ui: str | None = None, lang: str | None = None,
          port: int = 8765, open_browser: bool = True, read: Callable[[str], str] = input,
          out: Callable[[str], None] = print) -> int:
    """Ask for the interface language, the language model and the interface, then open it.

    Choices given as arguments are not asked. The last choices are remembered in
    runs/.chats/start.json and offered as defaults.
    """
    try:
        return _start(root, backend, ui, lang, port, open_browser, read, out)
    except (EOFError, KeyboardInterrupt):
        out("")
        return 1


def _start(root, backend, ui, lang, port, open_browser, read, out) -> int:
    from .chat import Assistant, Runs, serve

    saved_path = root / "runs" / ".chats" / "start.json"
    try:
        saved = json.loads(saved_path.read_text())
    except (OSError, ValueError):
        saved = {}

    if lang is None:
        lang = _choose("界面语言 / Language:", [("zh", "中文"), ("en", "English")], saved.get("lang"), read, out)
    zh = lang == "zh"

    if backend is None:
        found = backends.available()
        if not found:
            out("没有找到语言模型。在 MDAgent1 文件夹里运行 bash scripts/install.sh，它会安装并登录 Claude Code 或 Codex。\n"
                "不用语言模型也可以直接用命令行：python -m mdagent run --help" if zh else
                "No language model found. In the MDAgent1 folder, run: bash scripts/install.sh. It installs and logs in "
                "Claude Code or Codex.\n"
                "You can also use the command line without a language model: python -m mdagent run --help")
            return 2
        if len(found) == 1:
            backend = found[0]
            out(f"语言模型：{BACKEND_NAMES[backend]}（这台电脑上只找到这一个）" if zh else
                f"Language model: {BACKEND_NAMES[backend]} (the only one found on this computer)")
    if backend is None:
        backend = _choose("用哪个语言模型？它只把你的话填进设置表，不执行任何命令。" if zh else
                          "Which language model? It only fills in the settings form and runs nothing.",
                          [(b, BACKEND_NAMES[b]) for b in found], saved.get("backend"), read, out)

    login = backends.login_problem(backend)
    if login:
        out(f"{BACKEND_NAMES.get(backend, backend)} 还没有登录。请先运行：{login}\n"
            "它会显示一个链接，在任何一台电脑或手机的浏览器里打开、登录，需要时把代码贴回终端。然后再运行 mdagent start。" if zh else
            f"{BACKEND_NAMES.get(backend, backend)} is not logged in. First run: {login}\n"
            "It shows a link: open it in a browser on any computer or phone, sign in, and paste the code back if "
            "asked. Then run mdagent start again.")
        return 2

    if ui is None:
        ui = _choose("在哪里对话？" if zh else "Where do you want to talk?",
                     [("web", "网页（浏览器打开 http://localhost:%d）" % port if zh else
                       "Web page (http://localhost:%d in the browser)" % port),
                      ("terminal", "终端（就在这个窗口里）" if zh else "Terminal (in this window)")],
                     saved.get("ui"), read, out)

    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_text(json.dumps({"lang": lang, "backend": backend, "ui": ui}))
    if ui == "web":
        serve(root, backend, port, open_browser=open_browser)
        return 0
    return run_terminal(Assistant(root=root, backend=backend, runs=Runs(root), ui="terminal"), lang, read, out)
