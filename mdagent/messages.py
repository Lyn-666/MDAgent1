"""Messages from a run to the user.

A run writes a message when the user should know something: for example that
plain MD makes no progress and another method would help, or that the results
are ready. Each message is one JSON line in runs/<name>/messages.jsonl, with the
text in Chinese and English. The chat (web page and terminal) shows new messages
in the conversation that started the run. The run never waits for an answer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

FILE = "messages.jsonl"


def post(run_dir: Path, zh: str, en: str, kind: str = "info") -> None:
    """Add a message. kind: "info", "suggestion" (another method would help) or "done" (results ready)."""
    line = json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "zh": zh, "en": en},
                      ensure_ascii=False)
    with open(Path(run_dir) / FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass  # A line that is still being written.
    return out
