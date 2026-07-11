"""JSONL run logger for agent invocations.

Every agent call is appended as one JSON line: decision date, model, prompt version, the
raw and validated output, latency, and any validation error. This gives an auditable,
replayable trail (VRI Week-2 "logging"; supports the Week-4 two-pass evaluation).
"""

from __future__ import annotations

import json
from pathlib import Path


class RunLogger:
    """Append-only JSONL logger. ``latency`` is supplied by the caller (scripts are not
    permitted to read the wall clock inside deterministic/replayable runs)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_invocation(self, *, agent: str, prompt_version: str, as_of: str,
                       raw_output, assessment: dict | None,
                       error: str | None = None, latency_s: float | None = None,
                       extra: dict | None = None) -> None:
        rec = {
            "agent": agent,
            "prompt_version": prompt_version,
            "as_of": as_of,
            "raw_output": raw_output,
            "assessment": assessment,
            "error": error,
            "latency_s": latency_s,
        }
        if extra:
            rec.update(extra)
        self.log(rec)

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
