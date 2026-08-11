"""Cortex's own event log.

One JSONL file per Pondside day under ``$XDG_STATE_HOME/cortex/logs``. A day is a
``cat`` and rotation is a deletion.

Cortex logs an operation even when something upstream already logged it. The inference
gateway keeps its own record of every chat and embedding call, but that record belongs
to the gateway: move inference to another host, or drop the gateway, and it goes. This
log is Cortex's, and it must stay complete on its own.

This is an event log rather than a diagnostic one. Every line carries ``at`` and
``event``; the rest belongs to the event, and fields may be added at any time, so
readers must tolerate lines that lack them. Diagnostics go to stderr, which Claude Code
surfaces in the transcript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pendulum

from cortex.clock import pondside_day
from cortex.config import state_home


def logs_root() -> Path:
    """Return the directory holding the daily log files."""
    return state_home() / "cortex" / "logs"


def write(event: str, **fields: Any) -> None:
    """Append one event to today's log.

    A log that cannot be written must not take the caller down with it, so failures are
    reported on stderr and swallowed. Recollection is enrichment; losing its record is
    cheaper than losing the turn.

    Args:
        event: What happened, for example ``"recollection"``.
        **fields: Everything else about it. Must be JSON-serialisable.
    """
    line = json.dumps(
        {"at": pendulum.now().isoformat(), "event": event, **fields}, ensure_ascii=False
    )
    try:
        root = logs_root()
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"{pondside_day()}.jsonl").open("a", encoding="utf-8") as handle:
            _ = handle.write(line + "\n")
    except OSError as error:
        print(f"cortex: could not write to the event log: {error}", file=sys.stderr)
