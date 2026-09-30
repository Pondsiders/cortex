"""Reading memory Markdown files off disk.

A memory is ``$CORTEX_ROOT/memories/YYYY-MM-DD/<id>.md``: YAML frontmatter carrying
``created`` and an optional ``forgotten``, then the body. Attachments live in a
sidecar ``<id>/`` folder and are referenced from the body by wikilink embed syntax.

The disk is authoritative. Everything here reads; nothing here writes.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import frontmatter
import pendulum

_MEMORY_NAME = re.compile(r"^(\d+)\.md$")
_DAY_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class MemoryError_(Exception):
    """Raised when a file under the memories tree cannot be understood."""


@dataclass(frozen=True)
class Memory:
    """One memory, as it exists on disk."""

    id: int
    path: str
    created: datetime
    body: str
    forgotten: bool
    content_hash: str


def _hash(data: bytes) -> str:
    """Return the content hash of a memory file's raw bytes.

    Hashing the whole file — frontmatter included — is what makes a ``forgotten: true``
    edit detectable by an incremental reindex.
    """
    return hashlib.sha256(data).hexdigest()


def read(path: Path, root: Path) -> Memory:
    """Read and parse a single memory file.

    Args:
        path: Absolute path to the ``<id>.md`` file.
        root: The ``CORTEX_ROOT``, used to compute the stored relative path.

    Returns:
        The parsed memory.

    Raises:
        MemoryError_: If the filename, frontmatter, or ``created`` value is unusable.
    """
    match = _MEMORY_NAME.match(path.name)
    if match is None:
        msg = f"{path}: filename is not <id>.md"
        raise MemoryError_(msg)

    data = path.read_bytes()
    try:
        post = frontmatter.loads(data.decode("utf-8"))
    except Exception as exc:
        msg = f"{path}: could not parse frontmatter: {exc}"
        raise MemoryError_(msg) from exc

    raw_created = post.metadata.get("created")
    if raw_created is None:
        msg = f"{path}: frontmatter has no 'created'"
        raise MemoryError_(msg)
    try:
        if isinstance(raw_created, datetime):
            created = pendulum.instance(raw_created)
        else:
            created = pendulum.parse(str(raw_created))
    except Exception as exc:
        msg = f"{path}: unparseable 'created': {raw_created!r}"
        raise MemoryError_(msg) from exc
    if not isinstance(created, pendulum.DateTime):
        msg = f"{path}: 'created' is not a datetime: {raw_created!r}"
        raise MemoryError_(msg)
    if created.tzinfo is None:
        msg = f"{path}: 'created' has no timezone offset"
        raise MemoryError_(msg)

    forgotten = bool(post.metadata.get("forgotten", False))

    return Memory(
        id=int(match.group(1)),
        path=str(path.relative_to(root)),
        created=created,
        body=post.content,
        forgotten=forgotten,
        content_hash=_hash(data),
    )


def latest(memories_root: Path) -> Path | None:
    """Return the path of the highest-numbered memory on disk.

    Filenames only: nothing is opened and nothing is parsed, unlike :func:`discover`,
    which reads every memory there is. ``store`` uses it to pick the next id.

    The whole tree is scanned rather than today's day folder, so that nothing has to
    know about the 6 AM seam.

    Args:
        memories_root: The ``memories`` directory.

    Returns:
        The path to the largest ``<id>.md``, or None if the tree holds no memories.
    """
    if not memories_root.is_dir():
        return None

    best: tuple[int, Path] | None = None
    for day in os.scandir(memories_root):
        if not day.is_dir():
            continue
        for entry in os.scandir(day.path):
            match = _MEMORY_NAME.match(entry.name)
            if match is None:
                continue
            found = int(match.group(1))
            if best is None or found > best[0]:
                best = (found, Path(entry.path))
    return None if best is None else best[1]


@dataclass(frozen=True)
class Stat:
    """A memory file as the directory sees it: where it is, and whether it has moved."""

    id: int
    path: str
    mtime_ns: int
    size: int


def scan(memories_root: Path, root: Path) -> list[Stat]:
    """List every memory file with its size and modification time, ordered by id.

    Nothing is opened. This is what lets a sync skip the files that haven't changed:
    reading and parsing all of them cost 1.6 of the 2 seconds a store took at 21,000
    memories (#7), and a directory walk costs a few hundredths.

    Args:
        memories_root: The ``memories`` directory.
        root: The ``CORTEX_ROOT``, used to compute the stored relative path.

    Returns:
        One entry per memory file.

    Raises:
        MemoryError_: If a day folder or file name is malformed, or an id repeats.
    """
    if not memories_root.is_dir():
        msg = f"no memories directory at {memories_root}"
        raise MemoryError_(msg)

    # Relative paths are built as strings: pathlib per file was a third of the scan.
    prefix = memories_root.relative_to(root).as_posix()
    found: list[Stat] = []
    for day in sorted(memories_root.iterdir()):
        if day.name.startswith("."):
            continue
        if not day.is_dir():
            msg = f"{day}: unexpected file in the memories tree"
            raise MemoryError_(msg)
        if _DAY_NAME.match(day.name) is None:
            msg = f"{day}: day folder is not YYYY-MM-DD"
            raise MemoryError_(msg)
        for entry in os.scandir(day):
            if entry.is_dir() or entry.name.startswith("."):
                continue
            match = _MEMORY_NAME.match(entry.name)
            if match is None:
                msg = f"{entry.path}: filename is not <id>.md"
                raise MemoryError_(msg)
            stat = entry.stat()
            found.append(
                Stat(
                    id=int(match.group(1)),
                    path=f"{prefix}/{day.name}/{entry.name}",
                    mtime_ns=stat.st_mtime_ns,
                    size=stat.st_size,
                )
            )

    seen: dict[int, str] = {}
    for item in found:
        if item.id in seen:
            msg = f"duplicate memory id {item.id}: {seen[item.id]} and {item.path}"
            raise MemoryError_(msg)
        seen[item.id] = item.path

    return sorted(found, key=lambda item: item.id)


def discover(memories_root: Path, root: Path) -> Iterator[Memory]:
    """Walk the memories tree and yield every memory, read and parsed, ordered by id.

    Args:
        memories_root: The ``memories`` directory.
        root: The ``CORTEX_ROOT``.

    Yields:
        Each memory found on disk.

    Raises:
        MemoryError_: If a day folder or memory file is malformed.
    """
    for item in scan(memories_root, root):
        yield read(root / item.path, root)
