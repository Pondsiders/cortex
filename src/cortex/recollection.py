"""Involuntary recall: what a message reminds Alpha of, whether or not she asked.

Deliberate search is :mod:`cortex.search` — a question, an answer. This is the other
thing. It fires on every message, nobody chose it, and what it surfaces is the least
authoritative thing in the window rather than the most. The plumbing is the same
embeddings and the same cosine; the shape is the opposite.

Two paths produce a result set, and only one of them can fail:

*Cued.* The chat model decomposes the message into query strings, all of them are
embedded in one pass, and each takes the highest-scoring memory this session has not
already been shown.

*The Lagniappe.* One memory drawn uniformly at random. It answers nothing and is
related to nothing, which is the point: cosine recall is rich-get-richer, and a corpus
this size has memories that sit on nobody's nearest-neighbour list and are therefore
unreachable by similarity at any threshold, forever. A uniform draw is the only probe
with even reach.

That the Lagniappe needs no network is what makes it the floor as well as the garnish.
When the models are unreachable the cued path is abandoned and the stray goes out alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import final

import frontmatter
import numpy as np

from cortex import clock, log, seen
from cortex import index as index_module
from cortex.chat import Decomposer
from cortex.config import Settings
from cortex.embeddings import Embedder

BUDGET = 9990
"""Characters of everything the hook returns, header included. Claude Code caps a hook
at 10,000 and does not truncate past it: the whole thing is written to a file and
replaced with a path and a preview, so overrunning costs every memory rather than the
last one."""

HEADER = "cortex recollection hook output:"
"""Names the hook that produced the block. Alpha receives several anonymous context
blocks a turn and cannot otherwise tell them apart — or tell them from Jeffery."""

DEADLINE = 20.0
"""Seconds for the two network calls together. Claude Code allows the hook 30 and
discards the output of one that overruns, so the budget stops short of it."""


@final
@dataclass(frozen=True)
class Recollected:
    """One memory on its way back, and why it came."""

    id: int
    created: datetime
    body: str
    query: str | None
    score: float | None

    def block(self) -> str:
        """Render the memory as a ``## Memory #...`` block."""
        provenance = (
            ["- random memory"]
            if self.query is None or self.score is None
            else [f"- query: {self.query!r}", f"- score: {self.score:.2f}"]
        )
        return "\n".join(
            [
                f"## Memory #{self.id}",
                "",
                f"- {clock.pso8601(self.created)}",
                f"- {clock.age(self.created)}",
                *provenance,
                "",
                self.body,
            ]
        )


@final
@dataclass(frozen=True)
class Recollection:
    """Everything one firing of the hook produced."""

    memories: list[Recollected]
    queries: list[str]
    degraded: bool

    def context(self) -> str:
        """Assemble the blocks that fit the budget, most significant first.

        The list arrives in query order with the Lagniappe last, so overrunning drops
        the stray before it drops anything the message actually asked for. A single
        memory too large to fit at all is sliced rather than dropped, because returning
        nothing is worse than returning the beginning of something.

        The header is the first thing in the buffer rather than something added after
        the accounting, so the returned string is never longer than the budget.
        """
        if not self.memories:
            return ""
        parts = [HEADER]
        used = len(HEADER)
        for memory in self.memories:
            block = memory.block()
            if used + 2 + len(block) > BUDGET:
                if len(parts) == 1:
                    return f"{HEADER}\n\n{block}"[:BUDGET]
                break
            parts.append(block)
            used += 2 + len(block)
        return "\n\n".join(parts)


def _assign(scores: np.ndarray, queries: list[str]) -> dict[int, tuple[int, float]]:
    """Give each query its best still-unclaimed row.

    Walking the queries in turn would let whichever query the chat model happened to
    emit first take a contested memory and leave the loser with nothing, making the
    result depend on an ordering nobody chose. Instead every (query, row) pair competes
    on score.

    Only each query's top *n* rows can ever be needed, since at most *n-1* rivals can
    take one out from under it — so this sorts n² candidates rather than the whole
    matrix.

    Args:
        scores: A ``(queries, rows)`` similarity matrix, ineligible rows already -inf.
        queries: The query strings, for length only.

    Returns:
        Query index to (row, score), for the queries that got one.
    """
    n = len(queries)
    width = min(n, scores.shape[1])
    candidates = np.argpartition(scores, -width, axis=1)[:, -width:]

    pairs = sorted(
        (
            (float(scores[q, row]), q, int(row))
            for q in range(n)
            for row in candidates[q]
            if np.isfinite(scores[q, row])
        ),
        reverse=True,
    )

    taken: dict[int, tuple[int, float]] = {}
    claimed: set[int] = set()
    for score, q, row in pairs:
        if q in taken or row in claimed:
            continue
        taken[q] = (row, score)
        claimed.add(row)
        if len(taken) == n:
            break
    return taken


def _read(settings: Settings, entry: index_module.Entry) -> str | None:
    """Return a memory's body, or None if its file has gone."""
    path: Path = settings.cortex_root / entry.path
    if not path.is_file():
        return None
    return frontmatter.loads(path.read_text(encoding="utf-8")).content.strip()


def recollect(
    settings: Settings, *, prompt: str, session_id: str, deadline: float = DEADLINE
) -> Recollection:
    """Recall what a message brings to mind.

    Args:
        settings: Where the index lives and which models to ask.
        prompt: The user's message, verbatim.
        session_id: The harness's identifier for this session.
        deadline: Seconds allowed for the network calls.

    Returns:
        The memories, in the order they should be shown.

    Raises:
        index_module.IndexError_: If there is no index to search.
    """
    loaded = index_module.Index.load(
        settings.index_root, expect_model=settings.embedding_model
    )
    if loaded is None:
        msg = f"no index at {settings.index_root}; run cortex reindex"
        raise index_module.IndexError_(msg)

    ids = np.fromiter((entry.id for entry in loaded.entries), dtype=np.int64)
    mask = seen.Seen.load(session_id, int(ids.max(initial=0)))
    eligible = mask.filter(ids)

    queries: list[str] = []
    degraded = False
    found: list[Recollected] = []

    # A slash command carries no semantic content worth decomposing, but it is still a
    # turn, and the Lagniappe does not depend on what the message says.
    if not prompt.startswith("/"):
        try:
            queries, found = _cued(settings, prompt, loaded, eligible, deadline)
        # Any failure of the cued path degrades to the Lagniappe, which needs no
        # network of its own.
        except Exception as error:
            degraded = True
            log.write("recollection.degraded", session_id=session_id, error=str(error))

    for memory in found:
        eligible[np.flatnonzero(ids == memory.id)] = False

    stray = _lagniappe(settings, loaded, eligible)
    if stray is not None:
        found.append(stray)

    mask.mark([memory.id for memory in found])
    mask.save()
    return Recollection(memories=found, queries=queries, degraded=degraded)


def _cued(
    settings: Settings,
    prompt: str,
    loaded: index_module.Index,
    eligible: np.ndarray,
    deadline: float,
) -> tuple[list[str], list[Recollected]]:
    """Run the cued path: decompose, embed, and take the best row per query."""
    queries = Decomposer(settings, timeout=deadline / 2).decompose(prompt)
    if not queries:
        return [], []

    embedder = Embedder(settings, timeout=deadline / 2, max_retries=0)
    vectors = index_module.normalize(
        np.asarray(embedder.embed_queries(queries), dtype=np.float32)
    )

    scores = vectors @ np.asarray(loaded.vectors).T
    scores[:, ~eligible] = -np.inf

    found: list[Recollected] = []
    assigned = _assign(scores, queries)
    for q in range(len(queries)):
        if q not in assigned:
            continue
        row, score = assigned[q]
        entry = loaded.entries[row]
        body = _read(settings, entry)
        if body is None:
            continue
        found.append(
            Recollected(
                id=entry.id,
                created=entry.created,
                body=body,
                query=queries[q],
                score=score,
            )
        )
    return queries, found


def _lagniappe(
    settings: Settings, loaded: index_module.Index, eligible: np.ndarray
) -> Recollected | None:
    """Draw one eligible memory uniformly at random.

    Uniform on purpose. Weighting the draw — towards the old, the isolated, the
    high-scoring — would smuggle in a theory about which memories deserve resurfacing,
    and nobody has one.
    """
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return None
    entry = loaded.entries[int(np.random.default_rng().choice(rows))]
    body = _read(settings, entry)
    if body is None:
        return None
    return Recollected(
        id=entry.id, created=entry.created, body=body, query=None, score=None
    )
