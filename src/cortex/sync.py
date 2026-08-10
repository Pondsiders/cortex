"""Bringing the index back in sync with the Markdown on disk.

This is the only code in Cortex that turns a memory file into a vector. ``store`` writes
Markdown and then calls in here; nothing else embeds anything. That is deliberate — when
two code paths both produced vectors they disagreed, because one embedded what arrived
on stdin and the other embedded what the frontmatter parser handed back, and the same
memory got two different numbers depending on which one made it. One writer, one answer.

Unchanged memories keep the vectors they already have, so re-running after a failure
resumes instead of restarting, and a ``store`` costs one embedding rather than a
full rebuild.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cortex import index as index_module
from cortex import memories as memories_module
from cortex.config import Settings
from cortex.embeddings import BATCH_SIZE, CONCURRENCY, Embedder

CHECKPOINT_EVERY = 16
"""Batches between writes. Bounds how much embedding work a crash can cost."""


@dataclass(frozen=True)
class Result:
    """What a sync did."""

    total: int
    embedded: int
    reused: int
    dropped: int


def sync(
    settings: Settings,
    *,
    force: bool = False,
    batch_size: int = BATCH_SIZE,
    concurrency: int = CONCURRENCY,
    on_start: Callable[[int], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Result:
    """Rebuild the index so it matches the memory files on disk.

    Args:
        settings: Where the memories and the index live, and which models to use.
        force: Re-embed everything, ignoring content hashes.
        batch_size: Texts per embedding request.
        concurrency: Embedding requests in flight.
        on_start: Called once with the number of memories about to be embedded.
        on_progress: Called with each batch's size as batches land.

    Returns:
        Counts describing what happened.
    """
    root: Path = settings.index_root

    on_disk = list(
        memories_module.discover(settings.memories_root, settings.cortex_root)
    )
    live = [memory for memory in on_disk if not memory.forgotten]
    live_ids = {memory.id for memory in live}

    existing = index_module.Index.load(root, expect_model=settings.embedding_model)
    indexed_hashes = {} if existing is None else existing.hashes()
    indexed_rows = {} if existing is None else existing.rows()

    reusable_ids: set[int] = (
        set()
        if force
        else {m.id for m in live if indexed_hashes.get(m.id) == m.content_hash}
    )
    stale = [memory for memory in live if memory.id not in reusable_ids]
    dropped = len(set(indexed_rows) - live_ids)

    if not stale and dropped == 0 and existing is not None:
        return Result(total=len(live), embedded=0, reused=len(reusable_ids), dropped=0)

    embedder = Embedder(settings, batch_size=batch_size, concurrency=concurrency)
    dimensions = (
        int(existing.vectors.shape[1])
        if existing is not None
        else embedder.dimensions()
    )

    entries = [
        index_module.Entry(
            id=m.id, path=m.path, created=m.created, content_hash=m.content_hash
        )
        for m in live
    ]
    matrix = np.zeros((len(live), dimensions), dtype=np.float32)
    filled = np.zeros(len(live), dtype=bool)
    position_of = {memory.id: position for position, memory in enumerate(live)}

    if existing is not None:
        for memory_id in reusable_ids:
            position = position_of[memory_id]
            matrix[position] = existing.vectors[indexed_rows[memory_id]]
            filled[position] = True

    def checkpoint() -> None:
        rows = np.flatnonzero(filled)
        index_module.write(
            root,
            settings.embedding_model,
            [entries[row] for row in rows],
            matrix[rows],
        )

    if on_start is not None:
        on_start(len(stale))

    for batch_number, batch in enumerate(
        embedder.embed_documents([m.body for m in stale]), start=1
    ):
        for offset, vector in zip(batch.indices, batch.vectors, strict=True):
            position = position_of[stale[offset].id]
            matrix[position] = vector
            filled[position] = True
        if on_progress is not None:
            on_progress(len(batch.indices))
        if batch_number % CHECKPOINT_EVERY == 0:
            checkpoint()

    checkpoint()
    return Result(
        total=int(filled.sum()),
        embedded=len(stale),
        reused=len(reusable_ids),
        dropped=dropped,
    )
