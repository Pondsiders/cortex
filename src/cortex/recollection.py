"""Involuntary recall: what a message reminds Alpha of, whether or not she asked.

Deliberate search is :mod:`cortex.search` — a question, an answer. This is the other
thing. It fires on every message, nobody chose it, and what it surfaces is the least
authoritative thing in the window rather than the most. The plumbing is the same
embeddings and the same cosine; the shape is the opposite.

Three paths produce a result set, and only the first two can fail:

*Cued.* The chat model decomposes the message into query strings, all of them are
embedded in one pass, and each takes the highest-scoring memory this session has not
already been shown.

*Most topical.* The whole message is embedded beside the queries to measure every
memory's topicality. The single most topical unseen memory rides along when no query
claimed it. Seven of twelve long real messages had their most topical memory found by
no query: the queries took the message apart and lost what it was about (Oct 5 2026).

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

TOPIC_CHARS = 12_000
"""How much of the message goes into the topic vector. The embedding endpoint shares an
8,192-token context across its slots, and a message long enough to overflow it would
take the queries down with it; this is roughly four thousand tokens."""

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
    sigma: float | None = None
    """How far the score sits above this query's own mean over the whole corpus, in
    standard deviations. A raw cosine can't be read without it: 0.45 is a strong hit
    for a query whose corpus mean is 0.15 and nothing at all for one whose mean is 0.40
    (#5). Shown for the reader to judge, never used as a threshold."""
    topicality: float | None = None
    """How related the memory is to the whole message, in standard deviations above
    the message's own mean over the corpus. Zero is a memory drawn from a hat; high is
    on topic; negative is about something else. It differs from ``sigma`` on purpose:
    ``sigma`` grades the memory against the chat model's query, this grades it against
    what Jeffery actually said, so a high ``sigma`` with a low topicality is a query
    that wandered off. Random memories get one too."""
    topical: bool = False
    """Brought by the whole message rather than a query: the most topical memory, when
    no query claimed it."""

    def block(self) -> str:
        """Render the memory as a ``## Memory #...`` block."""
        # ruff reads the sigma as a confusable 'o'; it's display text, not a name.
        if self.topical:
            provenance = ["- most topical (whole message)"]
        elif self.query is None or self.score is None:
            provenance = ["- random memory"]
        else:
            scale = "" if self.sigma is None else f" ({self.sigma:+.1f}σ)"  # noqa: RUF001
            provenance = [
                f"- query: {self.query!r}",
                f"- score: {self.score:.2f}{scale}",
            ]
        if self.topicality is not None:
            provenance.append(f"- topicality: {self.topicality:+.1f}σ")  # noqa: RUF001
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
    """Everything one firing of the hook produced.

    ``memories`` is what Alpha is actually shown. Anything that was found but did not
    fit the budget is in ``dropped`` instead, and is still eligible later in the
    session.
    """

    memories: list[Recollected]
    queries: list[str]
    degraded: bool
    dropped: list[int]
    skipped: str | None
    """Why the message wasn't decomposed, when it wasn't."""

    def context(self) -> str:
        """Render the memories as one block of context, header first.

        :func:`_fit` has already decided what goes in, so everything here fits except
        possibly a lone memory too large for the budget by itself, which is sliced.
        """
        if not self.memories:
            return ""
        blocks = [memory.block() for memory in self.memories]
        return "\n\n".join([HEADER, *blocks])[:BUDGET]


def _fit(memories: list[Recollected]) -> list[Recollected]:
    """Keep the leading memories whose blocks fit the budget together.

    The list arrives in query order with the Lagniappe last, so overrunning drops the
    stray before it drops anything the message actually asked for. A single memory too
    large to fit at all is kept anyway and sliced when rendered, because returning
    nothing is worse than returning the beginning of something.

    This runs before anything is marked seen. When it ran afterwards, a memory cut for
    space was marked and never shown, and the log said it had been delivered (#1).

    The header is counted first, so the rendered string is never longer than the budget.
    """
    kept: list[Recollected] = []
    used = len(HEADER)
    for memory in memories:
        size = 2 + len(memory.block())
        if used + size > BUDGET:
            if not kept:
                kept.append(memory)
            break
        kept.append(memory)
        used += size
    return kept


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
    topic: np.ndarray | None = None

    skipped = _unspoken(prompt)
    if skipped is None:
        try:
            queries, found, topic = _cued(settings, prompt, loaded, eligible, deadline)
        # Any failure of the cued path degrades to the Lagniappe, which needs no
        # network of its own.
        except Exception as error:
            degraded = True
            log.write("recollection.degraded", session_id=session_id, error=str(error))

    for memory in found:
        eligible[np.flatnonzero(ids == memory.id)] = False

    stray = _lagniappe(settings, loaded, eligible, topic)
    if stray is not None:
        found.append(stray)

    shown = _fit(found)
    mask.mark([memory.id for memory in shown])
    mask.save()
    return Recollection(
        memories=shown,
        queries=queries,
        degraded=degraded,
        dropped=[memory.id for memory in found[len(shown) :]],
        skipped=skipped,
    )


def _unspoken(prompt: str) -> str | None:
    """Say why a prompt isn't worth decomposing, or None if it is.

    Three kinds of prompt get the Lagniappe and nothing else, because none of them is
    Jeffery saying something:

    - A slash command, which carries no content of its own.
    - Anything opening with ``[`` or ``<``. That's the memory bell, which opens
      ``[cron`` by our own convention, and every message the harness writes into the
      user's slot: ``<task-notification>``, ``<agent-message ...>`` and whatever it
      invents next. The hook's input has no field saying who sent a prompt, so the
      opening character is the only signal there is. Of 1,669 logged turns, 363
      opened this way and three were Jeffery (``<bash-input>``). Blanket rather than a
      list, so a new tag is quiet by default (#2).
    - Nothing at all.

    Returns:
        The prompt's opening word, for the log, or None to decompose it.
    """
    stripped = prompt.lstrip()
    if not stripped:
        return "empty"
    if stripped[0] in "/[<":
        return stripped.split(maxsplit=1)[0][:40]
    return None


def _cued(
    settings: Settings,
    prompt: str,
    loaded: index_module.Index,
    eligible: np.ndarray,
    deadline: float,
) -> tuple[list[str], list[Recollected], np.ndarray | None]:
    """Run the cued path: decompose, embed, and take the best row per query.

    The whole message is embedded alongside the queries, in the same request, as the
    topic vector. It retrieves nothing; it only measures every memory's topicality.

    Returns:
        The queries, the memories they found, and every row's topicality in standard
        deviations (None if the topic vector has no spread to measure against).
    """
    queries = Decomposer(settings, timeout=deadline / 2).decompose(prompt)

    embedder = Embedder(settings, timeout=deadline / 2, max_retries=0)
    vectors = index_module.normalize(
        np.asarray(
            embedder.embed_queries([*queries, prompt[:TOPIC_CHARS]]), dtype=np.float32
        )
    )

    scores = vectors @ np.asarray(loaded.vectors).T
    # Each row's scale is taken over the whole corpus before anything is masked, the
    # same way `cortex search` computes it, so the two read alike.
    baseline = scores.mean(axis=1)
    deviation = scores.std(axis=1)
    topic = (scores[-1] - baseline[-1]) / deviation[-1] if deviation[-1] else None
    scores = scores[:-1]
    if not queries:
        return [], [], topic
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
                sigma=(
                    float((score - baseline[q]) / deviation[q])
                    if deviation[q]
                    else None
                ),
                topicality=None if topic is None else float(topic[row]),
            )
        )
    topical = _most_topical(settings, loaded, eligible, topic, assigned)
    if topical is not None:
        found.append(topical)
    return queries, found, topic


def _most_topical(
    settings: Settings,
    loaded: index_module.Index,
    eligible: np.ndarray,
    topic: np.ndarray | None,
    assigned: dict[int, tuple[int, float]],
) -> Recollected | None:
    """The most topical unseen memory, unless a query already claimed it.

    Only reached when the message decomposed into at least one query, so an
    acknowledgement with no topic of its own doesn't get its nearest noise promoted.
    """
    if topic is None:
        return None
    ranked = np.where(eligible, topic, -np.inf)
    row = int(np.argmax(ranked))
    if not np.isfinite(ranked[row]) or row in {r for r, _ in assigned.values()}:
        return None
    entry = loaded.entries[row]
    body = _read(settings, entry)
    if body is None:
        return None
    return Recollected(
        id=entry.id,
        created=entry.created,
        body=body,
        query=None,
        score=None,
        topicality=float(topic[row]),
        topical=True,
    )


def _lagniappe(
    settings: Settings,
    loaded: index_module.Index,
    eligible: np.ndarray,
    topic: np.ndarray | None = None,
) -> Recollected | None:
    """Draw one eligible memory uniformly at random.

    Uniform on purpose. Weighting the draw — towards the old, the isolated, the
    high-scoring — would smuggle in a theory about which memories deserve resurfacing,
    and nobody has one. Its topicality is measured after the draw and reported, never
    used to choose: it's there so a stray that happens to fit can be called luck out
    loud, and one that doesn't can be enjoyed as weather.
    """
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return None
    row = int(np.random.default_rng().choice(rows))
    entry = loaded.entries[row]
    body = _read(settings, entry)
    if body is None:
        return None
    return Recollected(
        id=entry.id,
        created=entry.created,
        body=body,
        query=None,
        score=None,
        topicality=None if topic is None else float(topic[row]),
    )
