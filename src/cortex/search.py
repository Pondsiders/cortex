"""Deliberate search over the memory index.

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

from dataclasses import dataclass
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
    loaded = index_module.Index.load(
        settings.index_root, expect_model=settings.embedding_model
    )
    if loaded is None:
        msg = f"no index at {settings.index_root}; run cortex reindex"
        raise index_module.IndexError_(msg)

    vector = np.asarray(Embedder(settings).embed_query(query), dtype=np.float32)
    scores = np.asarray(loaded.vectors) @ vector

    baseline = float(scores.mean())
    deviation = float(scores.std())
    top = np.argsort(scores)[::-1][:limit]

    hits: list[Hit] = []
    for row in top:
        entry = loaded.entries[int(row)]
        path = settings.cortex_root / entry.path
        if not path.is_file():
            msg = f"{entry.path} is indexed but missing from disk; run cortex reindex"
            raise index_module.IndexError_(msg)
        score = float(scores[row])
        hits.append(
            Hit(
                id=entry.id,
                path=path,
                created=entry.created,
                score=score,
                sigma=(score - baseline) / deviation if deviation else 0.0,
                body=frontmatter.loads(path.read_text(encoding="utf-8")).content,
            )
        )

    return Results(
        corpus=len(loaded.entries),
        baseline=baseline,
        deviation=deviation,
        hits=hits,
    )
