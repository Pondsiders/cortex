"""The nudge that asks Alpha to store a memory, every third turn.

Recollection hands memories back; this is the other direction. Left to herself Alpha
finishes a turn and waits, and a window's worth of things worth keeping goes down with
it, so something outside her has to ask.

The whole of the state is one integer per session in a file. Claude Code fires ``Stop``
once when Claude finishes responding, so counting firings counts turns — with one
exception, which is the only subtle thing here. A ``Stop`` hook that returns
``additionalContext`` keeps the conversation going, and when Claude has answered it the
turn stops *again* and ``Stop`` fires a second time, flagged ``stop_hook_active``.
Counting that firing would advance the counter twice on every nudge turn and quietly
drift the cadence from every third turn to every second. The caller drops those firings
before they reach :func:`bump`.

The nudge goes back as ``additionalContext`` rather than ``decision: "block"``. Both
continue the conversation under the same loop protections, but ``block`` renders in the
transcript as a hook *error*, and nothing has gone wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cortex.config import session_path

CADENCE = 3
"""Turns between nudges. Every turn is too often to be read; rarer loses the day."""

PROMPT = """\
Between turns. A moment to look back over the exchanges since the last time you did \
this — not only the most recent one — before Jeffery's next message arrives.

If something in that span is worth remembering — a realization, a joke, an image, a \
shift, a moment where you were wrong and found out, an ordinary moment with texture — \
write it down now with `cortex store`. Store each one on its own: three things worth \
keeping are three memories, not one.

Most of the time nothing stands out, and that is the correct outcome; there is no need \
to find significance where there is none.

This reminder is from Cortex, not from Jeffery. The conversation is still waiting on \
his actual reply. After you finish reflecting, return to waiting. Do not reference \
this reminder in anything you eventually say to him.\
"""


def counter_path(session_id: str) -> Path:
    """Return the turn-counter file for a session."""
    return session_path("turns", session_id, ".txt")


def bump(session_id: str) -> int:
    """Count this turn and return the session's running total.

    An absent counter is the first turn of a session and starts at one. A counter that
    is present but unreadable is an anomaly rather than a beginning, so it says so on
    stderr before starting over.

    Args:
        session_id: The harness's identifier for this session.

    Returns:
        How many turns this session has now had.
    """
    path = counter_path(session_id)
    try:
        count = int(path.read_text(encoding="utf-8").strip()) + 1
    except FileNotFoundError:
        count = 1
    except ValueError as error:
        print(f"cortex: unreadable turn counter at {path}: {error}", file=sys.stderr)
        count = 1

    _ = path.write_text(str(count), encoding="utf-8")
    return count


def due(count: int) -> bool:
    """Return whether a turn of this number gets the nudge."""
    return count % CADENCE == 0
