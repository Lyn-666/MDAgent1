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
from pathlib import Path
from typing import Callable

from . import backends

TEXT = {
    "zh": {
        "hello": "MDAgent 终端对话（语言模型：{backend}）。用一句话说要跑什么，按 Enter 发送。输入 /help 查看命令。",
        "help": "命令（不用语言模型，不花 token）：\n"
                "  /runs  列出这个项目的运行：状态、最近的进度和结果文件夹\n"
                "  /new   开始一个新对话（不同的案例用不同的对话）\n"
                "  /quit  退出（正在跑的模拟不会停）\n"
                "其他输入都发给语言模型，例如：“用 181L，看苯周围的口袋，开发测试，有问题也继续”。",
        "you": "你> ", "agent": "MDAgent> ", "thinking": "（思考中…）",
        "changed": "改动的设置：", "none": "无", "command": "命令：", "check": "结构检查报告：",
        "ask_start": "开始运行？", "ask_stop": "停止运行？", "ask_resume": "继续运行？", "yes_no": " [y/N] ",
        "not_done": "没有执行。", "new_chat": "已开始新对话。", "bye": "再见。正在跑的模拟会继续在后台运行。",
        "plain_md": "普通 MD 每条 {ns:g} ns", "no_runs": "还没有运行。", "states": {"running": "运行中", "finished": "已完成", "stopped": "已停止",
                                            "failed": "出错"},
        "folder": "结果文件在电脑“{host}”的 {folder}（{n} 个文件）",
    },
    "en": {
        "hello": "MDAgent in the terminal (language model: {backend}). Say in one sentence what to run, then press "
                 "Enter. Type /help for commands.",
        "help": "Commands (no language model, no tokens):\n"
                "  /runs  list the runs of this project: state, latest progress and result folder\n"
                "  /new   start a new conversation (use one conversation per case)\n"
                "  /quit  leave (running simulations go on)\n"
                "Everything else goes to the language model, for example: \"use 181L, the pocket around benzene, "
                "development test, go on despite problems\".",
        "you": "you> ", "agent": "MDAgent> ", "thinking": "(thinking…)",
        "changed": "Changed settings: ", "none": "none", "command": "Command: ", "check": "Structure check report:",
        "ask_start": "Start the run?", "ask_stop": "Stop the run?", "ask_resume": "Continue the run?",
        "yes_no": " [y/N] ", "not_done": "Nothing was done.", "new_chat": "Started a new conversation.",
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
    if card["kind"] == "start":
        out(T["changed"] + (", ".join(card["changed"]) if card["changed"] else T["none"]))
        if card.get("check"):
            out(T["check"])
            out("  " + card["check"].replace("\n", "\n  "))
        out(T["command"] + card["command"])


def run_terminal(assistant, lang: str = "zh", read: Callable[[str], str] = input,
                 out: Callable[[str], None] = print) -> int:
    """Talk with the assistant in the terminal until /quit, Ctrl+D or Ctrl+C."""
    T = TEXT[lang]
    if read is input:
        _enable_line_editing()
    out(T["hello"].format(backend=BACKEND_NAMES.get(assistant.backend, assistant.backend)))
    chat_id = None
    try:
        while True:
            text = read("\n" + T["you"]).strip()
            if not text:
                continue
            if text in ("/quit", "/exit", "/q"):
                break
            if text == "/help":
                out(T["help"])
                continue
            if text == "/runs":
                show_runs(assistant, lang, out)
                continue
            if text == "/new":
                chat_id = None
                out(T["new_chat"])
                continue
            out(T["thinking"])
            r = assistant.message(text, lang, chat_id)
            chat_id = r["chat"]
            out(T["agent"] + r["reply"])
            card = r.get("card")
            if card:
                show_card(card, lang, out)
                answer = read(T[f"ask_{card['kind']}"] + T["yes_no"]).strip().lower()
                if answer in ("y", "yes", "是", "好", "开始", "继续", "停止"):
                    out(T["agent"] + assistant.confirm(card["id"], lang))
                else:
                    assistant.pending.pop(card["id"], None)
                    out(T["not_done"])
    except (EOFError, KeyboardInterrupt):
        out("")
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
            out("没有找到语言模型。请先安装并登录 Claude Code（claude）或 Codex（codex）。\n"
                "不用语言模型也可以直接用命令行：python -m mdagent run --help" if zh else
                "No language model found. Install and log in to Claude Code (claude) or Codex (codex) first.\n"
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
