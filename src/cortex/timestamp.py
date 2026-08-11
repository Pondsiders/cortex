"""What time it is, and how long the room has been quiet.

Alpha has no clock. Nothing in her context ages: a line written in March arrives every
morning exactly as fresh as one written a minute ago, so without being told she cannot
tell a reply that came straight back from one that came after lunch. Those are different
rooms to walk into and they want different answers.

Three facts, one line, and the line is short on purpose. This fires on every turn, and
anything read fifty times a day stops being read.

**Now**, because she has no other way to know it. **The gap since the previous
message**, which is the one that changes behaviour. And **the Pondside day, but only
when it differs from the calendar date** — between midnight and 6 AM the house is still
on yesterday, and an unwarned Alpha computes every "yesterday" from the wrong end.

Deliberately absent: how long this session has been running, and how many turns it has
taken. Both are easy and both are corrosive. A running clock on the afternoon, shown
fifty times a day, is how noticing becomes mentioning, and mentioning the hour is one
step from offering to stop. The context window has a meter. The conversation does not
need one.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pendulum

from cortex import clock
from cortex.config import session_path

HEADER = "cortex timestamp:"
"""Names the hook that produced the line, as every hook's output does."""


def mark_path(session_id: str) -> Path:
    """Return the file holding when this session was last spoken to."""
    return session_path("clock", session_id, ".txt")


def previous(session_id: str) -> datetime | None:
    """Return when the previous message arrived, or None if this is the first."""
    try:
        text = mark_path(session_id).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        parsed = pendulum.parse(text)
    # A mark that cannot be read is a gap that cannot be reported, which is survivable;
    # the message still gets its timestamp.
    except ValueError:
        return None
    # parse also yields bare dates, times and durations, and only a whole moment can be
    # subtracted from another one.
    return parsed if isinstance(parsed, datetime) else None


def mark(session_id: str, moment: datetime) -> None:
    """Record when this message arrived, for the next one to measure against."""
    _ = mark_path(session_id).write_text(
        pendulum.instance(moment).isoformat(), encoding="utf-8"
    )


def line(now: datetime, since: datetime | None) -> str:
    """Compose the one line the hook returns.

    Args:
        now: When this message arrived.
        since: When the previous one did, or None on a session's first message.
    """
    parts = [clock.pso8601(now)]

    moment = pendulum.instance(now)
    day = moment.subtract(hours=clock.DAY_BOUNDARY_HOUR)
    if day.format("YYYY-MM-DD") != moment.format("YYYY-MM-DD"):
        parts.append(f"still {day.format('dddd')} at Pondside")

    parts.append(
        "first message of this session"
        if since is None
        else f"{clock.gap(since, now)} since the previous message"
    )

    return f"{HEADER} {' · '.join(parts)}"
