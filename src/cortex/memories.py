"""Reading memory Markdown files off disk.

A memory is ``$CORTEX_ROOT/memories/YYYY-MM-DD/<id>.md``: YAML frontmatter carrying
``created`` and an optional ``forgotten``, then the body. Attachments live in a
sidecar ``<id>/`` folder and are referenced from the body by wikilink embed syntax.

The disk is authoritative. Everything here reads; nothing here writes.
"""

from __future__ import annotations

import hashlib
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


def discover(memories_root: Path, root: Path) -> Iterator[Memory]:
    """Walk the memories tree and yield every memory, ordered by id.

    Args:
        memories_root: The ``memories`` directory.
        root: The ``CORTEX_ROOT``.

    Yields:
        Each memory found on disk.

    Raises:
        MemoryError_: If a day folder or memory file is malformed.
    """
    if not memories_root.is_dir():
        msg = f"no memories directory at {memories_root}"
        raise MemoryError_(msg)

    found: list[Memory] = []
    for day in sorted(memories_root.iterdir()):
        if day.name.startswith("."):
            continue
        if not day.is_dir():
            msg = f"{day}: unexpected file in the memories tree"
            raise MemoryError_(msg)
        if _DAY_NAME.match(day.name) is None:
            msg = f"{day}: day folder is not YYYY-MM-DD"
            raise MemoryError_(msg)
        for entry in sorted(day.iterdir()):
            if entry.is_dir() or entry.name.startswith("."):
                continue
            found.append(read(entry, root))

    seen: dict[int, str] = {}
    for memory in found:
        if memory.id in seen:
            msg = (
                f"duplicate memory id {memory.id}: {seen[memory.id]} and {memory.path}"
            )
            raise MemoryError_(msg)
        seen[memory.id] = memory.path

    yield from sorted(found, key=lambda m: m.id)
