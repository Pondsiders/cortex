"""Deliberate search over the memory index, by query text or by neighboring memory.

Every result is scored against the whole corpus and the top *k* are returned. There is
no relevance threshold, deliberately: a query's best hit can be statistically
indistinguishable from noise and still be exactly right. Measured on this corpus, the
query "the AVN 2005 award won by Famke Janssen" has a corpus-wide maximum of 0.376 —
below any fixed floor anyone would pick — and the memory sitting there is the correct
one. The maximum of nineteen thousand draws lands about four and a half sigmas out by
chance alone, so no function of the score distribution can separate that case from
noise. Only a reader can. So the scores are reported, in context, and the judgement is
left to whoever asked.

Bodies are read from the Markdown rather than cached in the index, so what gets printed
is always the file as it stands now.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import frontmatter
import numpy as np

from cortex import index as index_module
from cortex.config import Settings
from cortex.embeddings import Embedder

DEFAULT_LIMIT = 5


@dataclass(frozen=True)
class Hit:
    """One search result."""

    id: int
    path: Path
    created: datetime
    score: float
    sigma: float
    body: str


@dataclass(frozen=True)
class Results:
    """A whole search: the hits, and the scale they should be read against."""

    corpus: int
    baseline: float
    deviation: float
    hits: list[Hit]
    source: str | None = None
    """The body of the memory whose neighbors these are, when there is one."""


def search(settings: Settings, query: str, limit: int = DEFAULT_LIMIT) -> Results:
    """Find the memories most similar to a query.

    Args:
        settings: Where the index lives and which model to query with.
        query: The search text.
        limit: How many results to return.

    Returns:
        The top hits, with the query's own corpus baseline for scale.

    Raises:
        index_module.IndexError_: If there is no index, or a hit's file is missing.
    """
    loaded = _load(settings)

    # Both sides unit-length, so the dot product in _rank is cosine similarity.
    vector = index_module.normalize(
        np.asarray([Embedder(settings).embed_query(query)], dtype=np.float32)
    )[0]
    return _rank(settings, loaded, vector, limit)


def similar(settings: Settings, memory_id: int, limit: int = DEFAULT_LIMIT) -> Results:
    """Find the memories most similar to one that is already stored.

    The memory's own row in the index is the query, so nothing is embedded and nothing
    leaves the machine: this works with the embedding endpoint down. The memory itself
    is left out of both the hits and the baseline, since it matches itself at exactly
    1.0 and that says nothing about its neighbors.

    Args:
        settings: Where the index and the memory files live.
        memory_id: The memory whose neighbors are wanted.
        limit: How many neighbors to return.

    Returns:
        The nearest neighbors, scaled against the rest of the corpus, with the source
        memory's own body attached.

    Raises:
        index_module.IndexError_: If there is no index, the memory is not in it, or a
            file the index names is missing.
    """
    loaded = _load(settings)

    row = loaded.rows().get(memory_id)
    if row is None:
        msg = f"memory #{memory_id} is not in the index; check the id, or reindex"
        raise index_module.IndexError_(msg)

    vector = np.asarray(loaded.vectors[row], dtype=np.float32)
    results = _rank(settings, loaded, vector, limit, skip=row)
    return replace(results, source=_body(settings, loaded.entries[row]))


def _load(settings: Settings) -> index_module.Index:
    loaded = index_module.Index.load(
        settings.index_root, expect_model=settings.embedding_model
    )
    if loaded is None:
        msg = f"no index at {settings.index_root}; run cortex reindex"
        raise index_module.IndexError_(msg)
    return loaded


def _body(settings: Settings, entry: index_module.Entry) -> str:
    path = settings.cortex_root / entry.path
    if not path.is_file():
        msg = f"{entry.path} is indexed but missing from disk; run cortex reindex"
        raise index_module.IndexError_(msg)
    return frontmatter.loads(path.read_text(encoding="utf-8")).content


def _rank(
    settings: Settings,
    loaded: index_module.Index,
    vector: np.ndarray,
    limit: int,
    *,
    skip: int | None = None,
) -> Results:
    """Score a unit vector against the corpus and read back the top hits.

    Args:
        settings: Where the memory files live.
        loaded: The index to score against.
        vector: A unit-length query vector.
        limit: How many hits to return.
        skip: A row to leave out of the hits and the baseline alike.
    """
    scores = np.asarray(loaded.vectors) @ vector

    counted = scores if skip is None else np.delete(scores, skip)
    baseline = float(counted.mean())
    deviation = float(counted.std())

    if skip is not None:
        scores[skip] = -np.inf
    top = np.argsort(scores)[::-1][:limit]

    hits: list[Hit] = []
    for row in top:
        entry = loaded.entries[int(row)]
        score = float(scores[row])
        hits.append(
            Hit(
                id=entry.id,
                path=settings.cortex_root / entry.path,
                created=entry.created,
                score=score,
                sigma=(score - baseline) / deviation if deviation else 0.0,
                body=_body(settings, entry),
            )
        )

    return Results(
        corpus=len(counted),
        baseline=baseline,
        deviation=deviation,
        hits=hits,
    )
