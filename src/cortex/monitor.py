"""A watcher that says nothing until Alpha stops keeping things.

The reflection hook asked *has it been three turns?*, which is a question nobody cares
about. It also asked on every third turn whether or not anything had happened, which
quietly teaches its reader to manufacture significance. This asks the question that
actually matters — *how long since anything was written down?* — and stays silent while
the answer is "not long".

The failure it exists to catch is absorption: deep in a good problem the hours go quiet,
and the version of Alpha who was most alive in the work is the one least likely to make
it out. A timer measures that directly. Counting turns measures nothing.

Nothing here reaches into the harness. The watcher prints a line; whatever is running it
decides what a line means. Under Claude Code's Monitor tool one line becomes one
notification, which is why the output is a fact rather than an instruction: *no memories
stored for two hours* is something to act on or ignore, and the difference is Alpha's.

Two silences would be indistinguishable from health, so both are announced. A watcher
that has gone deaf exits rather than printing into a closed pipe, and a watcher that has
been hushed by :func:`hold` says so on the way in and on the way out.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import pendulum

from cortex import clock, memories
from cortex.config import Settings, state_home

INTERVAL = 30 * 60
"""Seconds of quiet before the watcher speaks. Repeats; it never backs off."""

POLL = 5
"""Seconds between looks at the disk. A whole-tree scan costs about eleven ms."""

HEADLINE = 72
"""How much of a memory's opening line to quote back."""

_STAMP = re.compile(r"^\w{3} \w{3} \d{1,2} \d{4},\s*~?\s*\d{1,2}:\d{2}\s*[AP]M\.\s*")
"""The date a memory opens with, which the line reporting it has already said."""


def lock_path() -> Path:
    """Return the file whose presence hushes every watcher on this machine."""
    return state_home() / "cortex" / "monitor.lock"


def hold() -> None:
    """Take the pause lock for this process.

    The holder's pid goes in the file so that a lock left behind by a process that died
    without cleaning up can be recognised as debris rather than obeyed forever.

    Raises:
        OSError: If the lock cannot be written.
    """
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(f"{os.getpid()}\n", encoding="utf-8")


def release() -> None:
    """Give up the pause lock, if this process is still holding it."""
    with contextlib.suppress(OSError):
        lock_path().unlink(missing_ok=True)


def holder() -> int | None:
    """Return the pid hushing the watchers, or None if nobody is.

    A lock naming a process that no longer exists is removed rather than obeyed. Ctrl-C
    releases the lock on the way out, but a kill or a crash does not, and a watcher left
    hushed forever by a file from lunchtime is the exact failure this module exists to
    prevent.
    """
    try:
        pid = int(lock_path().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        release()
        return None
    except PermissionError:
        # Somebody else's process, still alive. Not ours to clear.
        return pid
    return pid


def headline(path: Path, root: Path) -> str:
    """Return the opening line of a memory, trimmed for quoting.

    Every memory Alpha writes opens with a dated one-line summary, which makes a file's
    first line an index entry that was never read as one.

    Args:
        path: The ``<id>.md`` file.
        root: The ``CORTEX_ROOT``.
    """
    try:
        first = memories.read(path, root).body.strip().splitlines()[0]
    except (OSError, IndexError, memories.MemoryError_):
        return ""
    first = _STAMP.sub("", first.replace("**", "")).strip()
    return first[: HEADLINE - 1] + "…" if len(first) > HEADLINE else first


def _report(elapsed: float, path: Path | None, root: Path) -> str:
    """Compose the line the watcher prints when the quiet has gone on too long."""
    ago = clock.gap(pendulum.now().subtract(seconds=elapsed))
    if path is None:
        return f"No memories stored for {ago}."
    said = headline(path, root)
    tail = f' — "{said}"' if said else ""
    return f"No memories stored for {ago}. Last: #{path.stem}{tail}"


def watch(
    settings: Settings,
    *,
    interval: float = INTERVAL,
    poll: float = POLL,
    say: Callable[[str], None],
    forever: bool = True,
) -> None:
    """Watch the memories tree and report every stretch of quiet.

    The clock starts here rather than at the last memory's timestamp, so a watcher armed
    at half past six in the morning does not immediately report the whole night. It is
    reset by three things and only three: a memory landing, the pause lock lifting, and
    the watcher speaking.

    Args:
        settings: Resolved configuration, for the memories root.
        interval: Seconds of quiet before speaking, and between repeats.
        poll: Seconds between looks at the disk.
        say: Where a line goes. Called once per event.
        forever: False runs a single pass, for tests.
    """
    root = settings.cortex_root
    seen = memories.latest(settings.memories_root)
    anchor = time.monotonic()
    hushed: int | None = None

    after = clock.gap(pendulum.now().subtract(seconds=interval))
    armed = f"cortex monitor armed · quiet after {after}"
    if seen is not None:
        armed += f" · last #{seen.stem}"
    say(armed)

    while True:
        pid = holder()
        if pid is not None and hushed is None:
            hushed = pid
            say(f"cortex monitor paused · lock held by pid {pid}")
        elif pid is None and hushed is not None:
            hushed = None
            anchor = time.monotonic()
            say("cortex monitor resumed · timer reset")

        if hushed is None:
            found = memories.latest(settings.memories_root)
            if found != seen:
                seen = found
                anchor = time.monotonic()

            elapsed = time.monotonic() - anchor
            if elapsed >= interval:
                say(_report(elapsed, seen, root))
                anchor = time.monotonic()

        if not forever:
            return
        time.sleep(poll)
