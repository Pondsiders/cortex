"""The on-disk index over the memory files.

Three files in one directory, all derived from Markdown and all disposable:

``vectors.npy``
    An ``(N, dimensions)`` float32 array. Row *i* belongs to entry *i*. Read back
    memory-mapped, so a search hands the bytes straight to BLAS without a copy.
``index.jsonl``
    One JSON object per row: id, path, created, content hash, and the file's size and
    modification time when it was read, which is how a sync knows it needn't read it
    again. Greppable.
``manifest.json``
    The one fact the other two cannot state about themselves — which embedding
    model produced the numbers — plus the row count, which is what makes a
    half-finished write detectable.

The index is never edited in place. It is rebuilt whole and moved into position with
``os.replace``, which is atomic, so a reader sees either the entire old index or the
entire new one. Rewriting all of it costs 34 ms at twenty thousand memories and 300 ms
at a hundred thousand, against a write path that runs a few dozen times a day; an
in-place append would save that and cost the atomicity, which is a bad trade.

``manifest.json`` is replaced last. If a rebuild dies mid-flight the manifest disagrees
with what is on disk, and loading refuses rather than serving a truncated corpus.

Writers take turns. Two syncs that both loaded the same index would each write it back
with only their own new memory, and their renames could interleave into three files
from two different indexes. :func:`locked` holds an exclusive lock on ``index.lock`` in
the index directory from load to rename, so one writer at a time. Readers take no lock;
the renames are what they rely on. The lock is ``flock``, which is only trustworthy on a
local filesystem, so the index should be written by one machine and only read by others.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import final

import numpy as np
import pendulum

VECTORS = "vectors.npy"
ENTRIES = "index.jsonl"
MANIFEST = "manifest.json"

LOCK = "index.lock"
LOCK_POLL = 0.05
"""Seconds between attempts while waiting for the lock with a timeout."""


class IndexError_(Exception):
    """Raised when the index on disk cannot be trusted."""


class Busy(Exception):
    """Raised when another writer still holds the index after the caller's timeout."""


@contextmanager
def locked(root: Path, *, timeout: float | None) -> Generator[None]:
    """Hold the index's write lock for the duration of the block.

    Args:
        root: The index directory; created if absent.
        timeout: Seconds to wait for the lock before giving up, or None to wait as
            long as it takes.

    Raises:
        Busy: If the lock is still held when ``timeout`` runs out.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / LOCK).open("a") as handle:
        if timeout is None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        else:
            _acquire_within(handle.fileno(), timeout, root)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _acquire_within(fd: int, timeout: float, root: Path) -> None:
    """Poll for the lock until it's ours or ``timeout`` seconds have passed."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            if time.monotonic() >= deadline:
                msg = f"another process is writing the index at {root}"
                raise Busy(msg) from error
            time.sleep(LOCK_POLL)
        else:
            return


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    Search is a plain dot product against these rows, which equals cosine similarity
    only if both sides are unit vectors. The embedding endpoint happens to return them
    that way, but nothing downstream checks, and a wrong-but-plausible score is worse
    than an error. Doing it here makes the property true rather than assumed, and costs
    nothing on vectors that already have it.

    Args:
        vectors: An ``(N, dimensions)`` array.

    Returns:
        The same array with every row scaled to length one. Zero rows are left alone.
    """
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors / np.where(lengths == 0.0, 1.0, lengths))


@dataclass(frozen=True)
class Entry:
    """One memory's place in the index."""

    id: int
    path: str
    created: datetime
    content_hash: str
    mtime_ns: int | None = None
    size: int | None = None
    """The file's modification time and size when it was last read. None in an index
    written before these were recorded, so the next sync reads everything once."""


def _entry_from_json(data: dict[str, object]) -> Entry:
    created = pendulum.parse(str(data["created"]))
    if not isinstance(created, pendulum.DateTime):
        msg = f"index entry {data.get('id')!r} has an unusable 'created'"
        raise IndexError_(msg)
    return Entry(
        id=int(str(data["id"])),
        path=str(data["path"]),
        created=created,
        content_hash=str(data["content_hash"]),
        mtime_ns=_optional_int(data.get("mtime_ns")),
        size=_optional_int(data.get("size")),
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))


@final
class Index:
    """A loaded index: the entries, and the matrix whose rows match them."""

    def __init__(
        self, entries: Sequence[Entry], vectors: np.ndarray, model: str
    ) -> None:
        """Bind entries to vectors. Prefer :meth:`load` over calling this directly."""
        self.entries: Sequence[Entry] = entries
        self.vectors: np.ndarray = vectors
        self.model: str = model

    @classmethod
    def load(cls, root: Path, *, expect_model: str | None = None) -> Index | None:
        """Read the index at ``root``, or return None if there isn't one yet.

        Args:
            root: The directory holding the three files.
            expect_model: If given, refuse an index built by a different embedding
                model. Serving vectors from the wrong model produces plausible
                nonsense rather than an error, so this is checked rather than assumed.

        Returns:
            The loaded index, or None if no manifest is present.

        Raises:
            IndexError_: If the three files disagree, or the model does not match.
        """
        manifest_path = root / MANIFEST
        if not manifest_path.is_file():
            return None

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = str(manifest["model"])
        if expect_model is not None and model != expect_model:
            msg = (
                f"index at {root} was built with {model!r}, "
                f"but {expect_model!r} is configured; run cortex reindex --force"
            )
            raise IndexError_(msg)

        vectors = np.load(root / VECTORS, mmap_mode="r")
        entries = [
            _entry_from_json(json.loads(line))
            for line in (root / ENTRIES).read_text(encoding="utf-8").splitlines()
            if line
        ]

        rows = int(manifest["rows"])
        if not (len(entries) == vectors.shape[0] == rows):
            msg = (
                f"index at {root} is inconsistent: manifest says {rows} rows, "
                f"{VECTORS} has {vectors.shape[0]}, {ENTRIES} has {len(entries)}; "
                f"run cortex reindex --force to rebuild it"
            )
            raise IndexError_(msg)

        return cls(entries, vectors, model)

    def hashes(self) -> dict[int, str]:
        """Return the content hash of every indexed memory, by id."""
        return {entry.id: entry.content_hash for entry in self.entries}

    def rows(self) -> dict[int, int]:
        """Return the row number of every indexed memory, by id."""
        return {entry.id: row for row, entry in enumerate(self.entries)}


def write(
    root: Path, model: str, entries: Sequence[Entry], vectors: np.ndarray
) -> None:
    """Replace the index at ``root`` with the given entries and vectors.

    Args:
        root: The directory to write into; created if absent.
        model: The embedding model that produced ``vectors``.
        entries: One entry per row, in row order.
        vectors: An ``(N, dimensions)`` float32 array.

    Raises:
        IndexError_: If the entries and vectors disagree on length.
    """
    if len(entries) != vectors.shape[0]:
        msg = f"{len(entries)} entries against {vectors.shape[0]} vectors"
        raise IndexError_(msg)

    root.mkdir(parents=True, exist_ok=True)

    # Private names, so a writer that ignores the lock or a crash mid-write can't leave
    # one process renaming another's half-written file.
    tmp_vectors = _staging(root, VECTORS)
    tmp_entries = _staging(root, ENTRIES)
    tmp_manifest = _staging(root, MANIFEST)
    try:
        with tmp_vectors.open("wb") as handle:
            np.save(handle, normalize(np.asarray(vectors, dtype=np.float32)))
            handle.flush()
            os.fsync(handle.fileno())

        with tmp_entries.open("w", encoding="utf-8") as handle:
            for entry in entries:
                _ = handle.write(
                    json.dumps(
                        {
                            "id": entry.id,
                            "path": entry.path,
                            "created": entry.created.isoformat(),
                            "content_hash": entry.content_hash,
                            "mtime_ns": entry.mtime_ns,
                            "size": entry.size,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

        with tmp_manifest.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": model,
                    "dimensions": int(vectors.shape[1]),
                    "rows": len(entries),
                    "written": pendulum.now().isoformat(),
                },
                handle,
                indent=2,
            )
            _ = handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # Manifest last: until it moves, the old index is still the live one.
        os.replace(tmp_vectors, root / VECTORS)
        os.replace(tmp_entries, root / ENTRIES)
        os.replace(tmp_manifest, root / MANIFEST)
    finally:
        for leftover in (tmp_vectors, tmp_entries, tmp_manifest):
            leftover.unlink(missing_ok=True)


def _staging(root: Path, name: str) -> Path:
    """Create an empty, uniquely named staging file for ``name`` in ``root``."""
    fd, path = tempfile.mkstemp(dir=root, prefix=f".{name}.", suffix=".tmp")
    os.close(fd)
    return Path(path)
