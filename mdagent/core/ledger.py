"""Reproducible record of every decision: decisions.jsonl, one JSON object per decision point.

Each record holds the progress, the assessment of every estimand (estimate, uncertainty, checks,
signals), the hypotheses with their heuristic label, every candidate with value, cost and authorization,
the choice, the stop reason, the suggestions, and what was executed. From these records alone a reader
can see why the agent did what it did; a later run with the same inputs gives the same records (apart
from the time stamp, which a test clock can fix).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

SCHEMA = 1


class Ledger:
    def __init__(self, path: Path | None = None, clock: Callable[[], str] | None = None):
        self.path = Path(path) if path else None
        self.clock = clock or (lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
        self.records: list[dict] = []
        if self.path and self.path.exists():
            self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, record: dict) -> dict:
        record = {"schema": SCHEMA, "step": len(self.records), "time": self.clock(), **record}
        self.records.append(record)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    @property
    def last(self) -> dict | None:
        return self.records[-1] if self.records else None
