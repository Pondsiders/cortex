"""Involuntary recall: what a message reminds Alpha of, whether or not she asked.

Deliberate search is :mod:`cortex.search` — a question, an answer. This is the other
thing. It fires on every message, nobody chose it, and what it surfaces is the least
authoritative thing in the window rather than the most. The plumbing is the same
embeddings and the same cosine; the shape is the opposite.

Three paths produce a result set:

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

The first two run on what Jeffery says. The Lagniappe runs on everything else: the
memory bell, notices, hand-backs, slash commands (Oct 23 2026; before that it rode on
every prompt). Claude Code says where a prompt came from, so that's what decides.

When a model can't be reached, recollection raises. It used to fall back to the
Lagniappe, and in five weeks of logs the only thing that fallback ever caught was a bug
of our own, which it hid (Oct 23 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, final

import frontmatter
import numpy as np
import pendulum

from cortex import clock, seen
from cortex import index as index_module
from cortex.chat import Decomposer
from cortex.config import Settings
from cortex.embeddings import Embedder

SPOKEN = frozenset({"composer", "bridge"})
"""Prompt origins that are Jeffery talking: his Enter at the prompt, or his phone."""

UNSPOKEN = frozenset(
    {
        "auto-continuation",
        "channel",
        "coordinator",
        "observer",
        "observer-activity",
        "peer",
        "peer-send-message",
        "plugin",
        "projects-relay",
        "scheduled-trigger",
        "sdk",
        "slack-ping",
        "task-notification",
        "unclassified",
    }
)
"""Every other origin Claude Code names (``PromptOrigin`` in its mod types). The memory
bell is ``scheduled-trigger``; subagent hand-backs are ``task-notification``."""

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
    that wandered off. Random memories don't
    have one: they ride on prompts that are never embedded."""
    topical: bool = False
    """Brought by the whole message rather than a query: the most topical memory, when
    no query claimed it."""

    @property
    def label(self) -> str:
        """What brought the memory, in a word or a query: the pane's stripe heading."""
        if self.topical:
            return "MOST TOPICAL"
        if self.query is None:
            return "RANDOM ACCESS"
        return self.query

    def record(self) -> dict[str, Any]:
        """The memory as data, for whatever draws it."""
        return {
            "id": self.id,
            "created": pendulum.instance(self.created).isoformat(),
            "label": self.label,
            "sigma": self.sigma,
            "topicality": self.topicality,
            "body": self.body,
        }

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
    """Everything one recollection produced, in the order it should be shown."""

    memories: list[Recollected]
    queries: list[str]

    def blocks(self) -> list[str]:
        """Render each memory as its own block of context."""
        return [memory.block() for memory in self.memories]


def spoken(origin: str, *, command: bool) -> bool:
    """Say whether a prompt is Jeffery talking, from where Claude Code says it came.

    A slash command is typed but says nothing of its own, so it counts as unspoken.

    Raises:
        ValueError: If Claude Code has invented an origin since this was written.
    """
    if origin in SPOKEN:
        return not command
    if origin in UNSPOKEN:
        return False
    msg = f"unknown prompt origin {origin!r}; add it to SPOKEN or UNSPOKEN"
    raise ValueError(msg)


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
    settings: Settings,
    *,
    prompt: str,
    session_id: str,
    spoken: bool,
    deadline: float = DEADLINE,
) -> Recollection:
    """Recall what a message brings to mind.

    What Jeffery says is decomposed and searched, and gets no Lagniappe. Everything
    else (the bell, notices, hand-backs, slash commands) gets the Lagniappe alone.

    Args:
        settings: Where the index lives and which models to ask.
        prompt: The user's message, verbatim.
        session_id: The harness's identifier for this session.
        spoken: Whether Jeffery said it (:func:`spoken`).
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

    if spoken:
        queries, found = _cued(settings, prompt, loaded, eligible, deadline)
    else:
        stray = _lagniappe(settings, loaded, eligible)
        queries, found = [], [] if stray is None else [stray]

    mask.mark([memory.id for memory in found])
    mask.save()
    return Recollection(memories=found, queries=queries)


def _cued(
    settings: Settings,
    prompt: str,
    loaded: index_module.Index,
    eligible: np.ndarray,
    deadline: float,
) -> tuple[list[str], list[Recollected]]:
    """Run the cued path: decompose, embed, and take the best row per query.

    The whole message is embedded alongside the queries, in the same request, as the
    topic vector. It retrieves nothing; it only measures every memory's topicality.

    Returns:
        The queries and the memories they found.
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
        return [], []
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
    return queries, found


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
) -> Recollected | None:
    """Draw one eligible memory uniformly at random.

    Uniform on purpose. Weighting the draw — towards the old, the isolated, the
    high-scoring — would smuggle in a theory about which memories deserve resurfacing,
    and nobody has one. It carries no topicality: it only rides on prompts nobody
    said, and those are never embedded.
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
    )
