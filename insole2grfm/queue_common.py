"""Helpers and names shared by the overnight queue and its entry gate.

``overnight_queue`` (the queue) and ``auto_correction`` (its entry gate) both
import from here, so neither imports the other's module at load time:
``overnight_queue`` imports ``auto_correction`` and injects its gate into
``Queue``; ``auto_correction`` only sees the ``Queue`` instance it is called with.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Task set of the entry gate (overfit test and validation guard).
GATE_TASK_SET = "continuous"
# Model whose first-fold validation curve is the guard reference (plan "guard_reference").
GUARD_REFERENCE = "mlp"
# Auto-correction log under the results root (JSON Lines, append-only).
AUTO_CORRECTION_LOG = "auto_correction.jsonl"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, payload: object) -> None:
    """Atomic write, so a reader (morning report) never sees a half-written file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def last_value(command: list[str], flag: str) -> str | None:
    """Value of the last ``flag`` in ``command`` (argparse keeps the last one)."""
    value = None
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            value = command[index + 1]
        elif item.startswith(flag + "="):
            value = item.split("=", 1)[1]
    return value
