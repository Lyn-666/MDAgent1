"""Language model backends for the chat window.

All backends take a system text, a user text and a Pydantic model, and return
an instance of that model. The model only fills in the form; it runs nothing.

- "claude": Claude Code in print mode (`claude -p`). Uses the Claude login of
  Claude Code (a subscription works, no API key). All tools are switched off.
- "codex": OpenAI Codex CLI (`codex exec`). Uses the login saved by Codex.
  Read-only sandbox.
- "api": the Anthropic API (needs ANTHROPIC_API_KEY or `ant auth login`).

Each call is independent: the chat window sends the recent conversation with
every call, so no backend needs to remember anything.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

BACKENDS = ("claude", "codex", "api")
TIMEOUT_S = 300


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema in the strict form that Codex requires (and Claude Code accepts).

    Every object lists all its properties as required and forbids extra ones.
    Optional fields stay optional by allowing null. Titles and defaults are removed.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def fix(node):
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


def available() -> list[str]:
    found = [b for b, exe in (("claude", "claude"), ("codex", "codex")) if shutil.which(exe)]
    if os.environ.get("ANTHROPIC_API_KEY"):
        found.append("api")
    return found


def choose(backend: str) -> str:
    if backend != "auto":
        return backend
    found = available()
    if not found:
        raise ValueError("No language model backend found. Install Claude Code (`claude`) or Codex (`codex`), "
                         "or set ANTHROPIC_API_KEY.")
    return found[0]


def call(backend: str, system: str, user: str, model: type[BaseModel], runner=subprocess.run) -> BaseModel:
    if backend == "claude":
        return _call_claude(system, user, model, runner)
    if backend == "codex":
        return _call_codex(system, user, model, runner)
    if backend == "api":
        return _call_api(system, user, model)
    raise ValueError(f"Unknown backend {backend!r}. Choose from {BACKENDS}.")


def _call_claude(system, user, model, runner):
    cmd = ["claude", "-p", user, "--output-format", "json",
           "--json-schema", json.dumps(strict_schema(model)),
           "--tools", "", "--system-prompt", system, "--no-session-persistence"]
    # Run in an empty folder, so Claude Code does not load this project's CLAUDE.md.
    with tempfile.TemporaryDirectory() as tmp:
        done = runner(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=tmp)
    if done.returncode != 0:
        raise ValueError(f"Claude Code failed: {(done.stderr or done.stdout).strip()[:500]}")
    data = json.loads(done.stdout)
    if data.get("structured_output") is None:
        raise ValueError(f"Claude Code returned no structured answer: {str(data.get('result'))[:500]}")
    return model.model_validate(data["structured_output"])


def _call_codex(system, user, model, runner):
    with tempfile.TemporaryDirectory() as tmp:
        schema_path, out_path = Path(tmp) / "schema.json", Path(tmp) / "answer.json"
        schema_path.write_text(json.dumps(strict_schema(model)))
        # Codex exec has no separate system prompt option, so the instructions go first.
        prompt = f"{system}\n\n---\n\n{user}"
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
               "--output-schema", str(schema_path), "-o", str(out_path), prompt]
        done = runner(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=tmp)
        if done.returncode != 0:
            raise ValueError(f"Codex failed: {(done.stderr or done.stdout).strip()[:500]}")
        if not out_path.exists():
            raise ValueError("Codex wrote no answer file.")
        return model.model_validate_json(out_path.read_text())


def _call_api(system, user, model):
    import anthropic

    from .llm import MODEL

    response = anthropic.Anthropic().beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=model,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise ValueError(f"The API returned no answer (stop reason: {response.stop_reason}).")
    return response.parsed_output
