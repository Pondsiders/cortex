"""The per-session record of which memories have already been shown.

A boolean array, one element per memory id, true where the memory may still be
recollected this session. It lives beside the session in the system temp directory, so
it dies when the machine cleans up after itself, which is the right lifetime for
something scoped to a session.

**Keyed by id, not by index row.** Rows are not stable within a session: ``store``
rebuilds the whole index on every memory, and a sync's intermediate checkpoints publish
only the rows filled so far, so row *N* names one memory before a reindex and a
different one after. A row-keyed mask would go on suppressing the wrong memories with
nothing anywhere raising. Ids never move.

The mask never shrinks and is never reordered. Where the corpus has grown since it was
written it is padded with false, so a memory stored during a session is never
recollected in that session — which is correct, because it is already in that session's
context.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import final

import numpy as np

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def mask_path(session_id: str) -> Path:
    """Return the mask file for a session.

    The session id arrives from the harness as JSON, so it is scrubbed to characters
    that cannot walk out of the temp directory rather than trusted as a filename.
    """
    return (
        Path(tempfile.gettempdir()) / f"cortex-seen-{_UNSAFE.sub('_', session_id)}.npy"
    )


@final
class Seen:
    """Which memories a session may still be shown."""

    def __init__(self, path: Path, eligible: np.ndarray) -> None:
        """Bind a mask to the file it came from. Prefer :meth:`load`."""
        self.path: Path = path
        self.eligible: np.ndarray = eligible

    @classmethod
    def load(cls, session_id: str, largest_id: int) -> Seen:
        """Read a session's mask, creating or extending it to cover ``largest_id``.

        Args:
            session_id: The harness's identifier for this session.
            largest_id: The largest memory id currently in the index.

        Returns:
            The mask, sized to at least ``largest_id + 1``.
        """
        path = mask_path(session_id)
        wanted = largest_id + 1

        if path.is_file():
            eligible = np.asarray(np.load(path), dtype=bool)
            if len(eligible) < wanted:
                eligible = np.concatenate(
                    [eligible, np.zeros(wanted - len(eligible), dtype=bool)]
                )
        else:
            eligible = np.ones(wanted, dtype=bool)
            eligible[0] = False  # There is no memory id 0.

        return cls(path, eligible)

    def filter(self, ids: np.ndarray) -> np.ndarray:
        """Return a row filter for an index's ids, true where the row may be shown.

        Args:
            ids: The memory id of each index row, in row order.

        Returns:
            A boolean array the same length as ``ids``.
        """
        return self.eligible[ids]

    def mark(self, ids: list[int]) -> None:
        """Record that these memories have now been shown."""
        if ids:
            self.eligible[np.asarray(ids, dtype=np.int64)] = False

    def save(self) -> None:
        """Write the mask back."""
        np.save(self.path, self.eligible)
