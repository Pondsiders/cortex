"""Output that gives up rather than taking the command down with it.

A command with a side effect is not a filter. ``cat`` should die when nobody is reading
it; ``cortex store`` should not, because by the time anything is printed a memory has
been written and the index has not yet caught up with it. A reader that stops early —
``| head``, ``| grep -q``, quitting ``less`` — closes the pipe, and the next write
raises. That put three memories into an unindexed limbo before anyone noticed, and
noticing took a person asking.

So a broken stream is treated as an audience that left rather than as an error. The
first failed write mutes that stream for the life of the process and everything after it
is dropped in silence. Muting is per stream: a progress bar that loses stderr must also
silence the status lines sharing it.

Every write is flushed as it is made. Buffering a stream that later breaks leaves the
interpreter to find out during shutdown, which it reports as an ignored exception — a
frightening message about output nobody was reading.

This is the near half of the guarantee. It cannot help with a kill, so the far half is
that the index is a cache: whatever drifts, a reindex settles.

Muting is the right answer for a command that has already done something. It is the
wrong answer for one whose output *is* the product: a long-running watcher that goes
deaf keeps running, printing into a closed pipe, indistinguishable from a watcher with
nothing to report. That command is a filter and should die like one, so :func:`deaf`
lets it ask whether anyone is still there.
"""

from __future__ import annotations

import sys
from typing import TextIO, final

_muted: set[int] = set()


def _put(stream: TextIO, text: str) -> None:
    """Write to a stream unless it has already stopped listening."""
    if id(stream) in _muted:
        return
    try:
        _ = stream.write(text)
        stream.flush()
    # ValueError is a stream closed out from under us; BrokenPipeError is a reader that
    # walked away. Neither is the command's problem.
    except (BrokenPipeError, ValueError):
        _muted.add(id(stream))


def out(message: str = "") -> None:
    """Write a line of the command's actual output to stdout."""
    _put(sys.stdout, message + "\n")


def say(message: str = "") -> None:
    """Write a line of commentary to stderr."""
    _put(sys.stderr, message + "\n")


def deaf(stream: TextIO) -> bool:
    """Report whether writes to a stream have stopped going anywhere.

    True once a write to it has failed, which is permanent for the life of the process.

    Args:
        stream: The stream to ask about, usually ``sys.stdout``.
    """
    return id(stream) in _muted


def progress_file() -> Sink:
    """Return a stderr stand-in for a progress bar to draw on."""
    return Sink(sys.stderr)


@final
class Sink:
    """The minimum file a progress bar needs, routed through this module."""

    def __init__(self, stream: TextIO) -> None:
        """Wrap a stream so writing to it can never raise."""
        self._stream: TextIO = stream

    def write(self, text: str) -> int:
        """Write, reporting success whether or not anything was written."""
        _put(self._stream, text)
        return len(text)

    def flush(self) -> None:
        """Do nothing; every write is flushed as it is made."""

    def isatty(self) -> bool:
        """Report whether the underlying stream is a terminal."""
        try:
            return self._stream.isatty()
        except ValueError:
            return False
