"""Cortex's circuit breaker: one switch, shared by every session.

When a model Cortex needs can't be reached, recollection trips the breaker instead of
raising on every prompt for as long as the outage lasts. Tripped, Cortex makes no model
calls and recollects nothing; each prompt carries one line saying so, so a fresh
context window learns it's off without having seen it trip. It stays off until somebody
runs ``cortex on``. Resetting is dumb on purpose, like the real thing: if the fault is
still there, the next prompt trips it again.

Only *absent* trips it. A model that answers wrongly (a 404, a 500, garbage) is a bug,
and bugs raise.

The switch is a file. Present means off, and it says when and why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import final

import pendulum

from cortex import clock
from cortex.config import state_home


def switch_path() -> Path:
    """Return the file whose presence means Cortex is off."""
    return state_home() / "cortex" / "off.json"


@final
@dataclass(frozen=True)
class Off:
    """When Cortex was switched off, and why."""

    at: datetime
    reason: str

    def line(self) -> str:
        """The line every prompt carries while Cortex is off."""
        when = pendulum.instance(self.at).format("h:mm A")
        return f"cortex: off since {when} ({self.reason}) · `cortex on` to reset"


def state() -> Off | None:
    """Return why Cortex is off, or None if it's on."""
    path = switch_path()
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    parsed = pendulum.parse(data["at"])
    assert isinstance(parsed, datetime)  # noqa: S101 — we wrote it with isoformat
    return Off(at=parsed, reason=str(data["reason"]))


def trip(reason: str) -> Off:
    """Switch Cortex off, saying why."""
    off = Off(at=pendulum.now(), reason=reason)
    path = switch_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps({"at": pendulum.instance(off.at).isoformat(), "reason": reason}),
        encoding="utf-8",
    )
    return off


def reset() -> None:
    """Switch Cortex back on."""
    switch_path().unlink(missing_ok=True)


def tripped_block(off: Off) -> str:
    """The one loud block a prompt gets on the turn the breaker trips."""
    reason = f"{off.reason[0].upper()}{off.reason[1:]}"
    return (
        f"cortex: TRIPPED at {clock.pso8601(off.at)}.\n\n"
        f"{reason}. Cortex is now off in every session and recollects nothing until "
        "someone runs `cortex on`. Every prompt will carry a one-line reminder until "
        "then."
    )
