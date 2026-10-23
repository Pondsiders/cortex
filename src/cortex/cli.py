"""The ``cortex`` command line."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import click
import pendulum
from tqdm import tqdm

from cortex import breaker, clock, console, log
from cortex import index as index_module
from cortex import memories as memories_module
from cortex import recollection as recollection_module
from cortex import search as search_module
from cortex import sync as sync_module
from cortex import timestamp as timestamp_module
from cortex.config import Settings
from cortex.embeddings import BATCH_SIZE, CONCURRENCY

STORE_LOCK_TIMEOUT = 10.0
"""Seconds a store waits for another sync before leaving its memory to the next."""


def _say(message: str) -> None:
    """Write a status line to stderr, safely alongside an active progress bar."""
    tqdm.write(message, file=console.progress_file())


def _run_sync(
    settings: Settings,
    *,
    force: bool,
    batch_size: int,
    concurrency: int,
    timeout: float | None = None,
) -> sync_module.Result:
    """Run a sync with a progress bar, returning its result."""
    bar: tqdm[Any] | None = None

    def on_start(count: int) -> None:
        nonlocal bar
        if count:
            _say(f"embedding {count} memories at {concurrency} x {batch_size}")
            bar = tqdm(
                total=count,
                unit="mem",
                smoothing=0.05,
                file=console.progress_file(),
            )

    def on_progress(landed: int) -> None:
        if bar is not None:
            _ = bar.update(landed)

    try:
        return sync_module.sync(
            settings,
            force=force,
            timeout=timeout,
            batch_size=batch_size,
            concurrency=concurrency,
            on_start=on_start,
            on_progress=on_progress,
        )
    finally:
        # pyright doesn't track the nonlocal assignment made inside on_start, so it
        # believes bar is still None here.
        if bar is not None:
            bar.close()  # pyright: ignore[reportUnreachable]


@click.group()
def cortex() -> None:
    """Alpha's memory: Markdown files, with a disposable index over them."""


@cortex.command()
def store() -> None:
    """Store a memory, reading its body from standard input.

    The memory file is written first and is the thing that matters; the index is then
    brought back in sync, which embeds the new memory and nothing else. This command
    never computes a vector itself.
    """
    body = sys.stdin.read()
    if not body.strip():
        raise click.ClickException("refusing to store an empty memory")

    settings = Settings()  # pyright: ignore[reportCallIssue]
    created = pendulum.now()
    folder = settings.memories_root / clock.pondside_day(created)
    folder.mkdir(parents=True, exist_ok=True)

    content = f"---\ncreated: {created.isoformat()}\n---\n\n{body.strip()}\n"
    path = _write_next(folder, settings.memories_root, content)
    console.out(str(path.relative_to(settings.cortex_root)))

    # The memory is safe on disk now. Another store's sync takes well under a second,
    # so wait out a few of those. A sync that holds the index longer than that is a
    # reindex, and isn't worth hanging for: whichever sync runs next scans the disk
    # and picks this file up.
    try:
        result = _run_sync(
            settings,
            force=False,
            timeout=STORE_LOCK_TIMEOUT,
            batch_size=BATCH_SIZE,
            concurrency=CONCURRENCY,
        )
    except index_module.Busy:
        _say("index busy; the next sync will pick this memory up")
        return
    _say(f"index: {result.total} memories, {result.embedded} embedded")


def _write_next(folder: Path, memories_root: Path, content: str) -> Path:
    """Create the next available ``<id>.md`` in ``folder`` and write ``content``.

    The id is one past the largest on disk. Two writers racing both scan, both guess the
    same id, and the loser's exclusive create fails and it tries the next one.
    """
    next_id = _max_id(memories_root) + 1
    while True:
        path = folder / f"{next_id}.md"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            next_id += 1
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            _ = handle.write(content)
        return path


def _max_id(memories_root: Path) -> int:
    """Return the largest memory id on disk, or zero if there are none."""
    path = memories_module.latest(memories_root)
    return 0 if path is None else int(path.stem)


def _print_hits(results: search_module.Results) -> None:
    for hit in results.hits:
        when = pendulum.instance(hit.created).format("ddd MMM D YYYY, h:mm A")
        # ruff reads the sigma as a confusable 'o'; it's display text, not a name.
        sigma = f"{hit.sigma:+.1f}σ"  # noqa: RUF001
        console.out(f"\n#{hit.id}  {hit.score:.4f}  {sigma}  {when}")
        console.out(f"{hit.path}\n")
        console.out(hit.body)


@cortex.command()
@click.option(
    "-k",
    "limit",
    default=search_module.DEFAULT_LIMIT,
    show_default=True,
    help="How many memories to return.",
)
def search(limit: int) -> None:
    """Search the memories, reading the query from standard input.

    Scores are reported rather than filtered. The header carries the query's own
    similarity to the whole corpus, which is what makes a raw cosine legible: 0.37
    against a 0.14 baseline is a strong hit, and against a 0.43 baseline it is nothing.
    """
    query = sys.stdin.read().strip()
    if not query:
        raise click.ClickException("refusing to search for nothing")

    settings = Settings()  # pyright: ignore[reportCallIssue]
    results = search_module.search(settings, query, limit=limit)

    scale = f"{results.baseline:.3f} ± {results.deviation:.3f}"
    console.out(f"{results.corpus:,} memories · this query's corpus baseline {scale}")
    _print_hits(results)


HEADLINE_WIDTH = 100


@cortex.command()
@click.argument("memory_id", metavar="ID", type=int)
@click.option(
    "-k",
    "limit",
    default=search_module.DEFAULT_LIMIT,
    show_default=True,
    help="How many neighbors to return.",
)
def similar(memory_id: int, limit: int) -> None:
    """Show the memories nearest to memory ID.

    The memory's stored vector is the query, so nothing is embedded and this works with
    the embedding endpoint down. The memory itself is never among its own neighbors.
    Scores read the same way they do for search.
    """
    settings = Settings()  # pyright: ignore[reportCallIssue]
    try:
        results = search_module.similar(settings, memory_id, limit=limit)
    except index_module.IndexError_ as error:
        raise click.ClickException(str(error)) from error

    first = (results.source or "").strip().splitlines()[:1]
    headline = first[0] if first else ""
    if len(headline) > HEADLINE_WIDTH:
        headline = headline[: HEADLINE_WIDTH - 1].rstrip() + "…"
    console.out(f'neighbors of #{memory_id} — "{headline}"')

    scale = f"{results.baseline:.3f} ± {results.deviation:.3f}"
    console.out(
        f"{results.corpus:,} other memories · #{memory_id}'s corpus baseline {scale}"
    )
    _print_hits(results)


@cortex.command()
@click.option(
    "--force", is_flag=True, help="Re-embed every memory, ignoring content hashes."
)
@click.option(
    "--batch-size",
    default=BATCH_SIZE,
    show_default=True,
    help="Texts per embedding request.",
)
@click.option(
    "--concurrency",
    default=CONCURRENCY,
    show_default=True,
    help="Embedding requests in flight.",
)
def reindex(force: bool, batch_size: int, concurrency: int) -> None:
    """Bring the index back in sync with the Markdown files on disk.

    Only memories whose file contents have changed are re-embedded, so a re-run after a
    failure resumes rather than starting over. Forgotten memories, and memories whose
    files have gone, drop out of the index entirely.
    """
    started = time.monotonic()
    settings = Settings()  # pyright: ignore[reportCallIssue]

    _say(f"root  {settings.cortex_root}")
    _say(f"index {settings.index_root}")

    result = _run_sync(
        settings, force=force, batch_size=batch_size, concurrency=concurrency
    )

    if result.embedded == 0 and result.dropped == 0:
        _say(f"index is current: {result.total} memories")
    else:
        counts = ", ".join(
            (
                f"{result.embedded} embedded",
                f"{result.reused} reused",
                f"{result.dropped} dropped",
            )
        )
        elapsed = f"{time.monotonic() - started:.1f}s"
        _say(f"indexed {result.total} memories in {elapsed} ({counts})")


@cortex.group()
def hook() -> None:
    """What the claude-code-cortex mod calls. Each reads JSON from standard input."""


@hook.command("prompt")
def hook_prompt() -> None:
    """Answer a prompt with the time and what it brings to mind.

    Reads one JSON object from standard input, written by the claude-code-cortex mod:
    ``text``, ``origin`` (Claude Code's ``origin.kind``), ``command`` (the slash
    command's name, or null) and ``session_id``. Prints one back: ``context``, the
    blocks the model reads beside the prompt (the timestamp line, then one per memory),
    ``memories``, the same memories as data for the Recollections pane, and ``off``,
    the breaker's state (null when Cortex is on).

    A model that can't be reached trips the breaker (:mod:`cortex.breaker`) instead of
    raising, and while it's tripped nothing is recollected. Everything else is
    unguarded: bad input, bad config or an origin nobody has seen raises, and the mod
    hands the traceback to Alpha in place of the context.
    """
    started = time.monotonic()
    packet: dict[str, Any] = json.loads(sys.stdin.read())
    prompt = str(packet["text"])
    origin = str(packet["origin"])
    command: str | None = packet["command"]
    session_id = str(packet["session_id"])

    stamp = _timestamp(session_id)
    spoken = recollection_module.spoken(origin, command=command is not None)

    off = breaker.state()
    if off is not None:
        _answer([stamp, off.line()], [], off)
        return

    settings = Settings()  # pyright: ignore[reportCallIssue]
    try:
        result = recollection_module.recollect(
            settings, prompt=prompt, session_id=session_id, spoken=spoken
        )
    except recollection_module.Unreachable as error:
        off = breaker.trip(error.reason)
        log.write("breaker.tripped", session_id=session_id, reason=error.reason)
        _answer([stamp, breaker.tripped_block(off)], [], off)
        return
    blocks = result.blocks()

    elapsed = int((time.monotonic() - started) * 1000)
    log.write(
        "recollection",
        session_id=session_id,
        ms=elapsed,
        prompt=prompt,
        queries=[
            {
                "q": m.query,
                "id": m.id,
                "score": round(m.score, 4),
                "sigma": None if m.sigma is None else round(m.sigma, 2),
                "topicality": _round(m.topicality),
            }
            for m in result.memories
            if m.query is not None and m.score is not None
        ],
        unanswered=[
            q for q in result.queries if q not in {m.query for m in result.memories}
        ],
        topical=next((m.id for m in result.memories if m.topical), None),
        topical_topicality=next(
            (_round(m.topicality) for m in result.memories if m.topical), None
        ),
        lagniappe=next(
            (m.id for m in result.memories if m.query is None and not m.topical), None
        ),
        lagniappe_topicality=next(
            (
                _round(m.topicality)
                for m in result.memories
                if m.query is None and not m.topical
            ),
            None,
        ),
        skipped=None if spoken else (f"/{command}" if command else origin),
        chars=sum(len(block) for block in blocks),
    )

    _answer([stamp, *blocks], [memory.record() for memory in result.memories], None)


def _answer(
    context: list[str], memories: list[dict[str, Any]], off: breaker.Off | None
) -> None:
    """Print what the mod reads back."""
    console.out(
        json.dumps(
            {
                "context": context,
                "memories": memories,
                "off": None
                if off is None
                else {
                    "at": pendulum.instance(off.at).isoformat(),
                    "reason": off.reason,
                },
            },
            ensure_ascii=False,
        )
    )


@cortex.command("on")
def switch_on() -> None:
    """Switch Cortex back on, after the breaker tripped or someone switched it off."""
    breaker.reset()
    console.out("on")


@cortex.command("off")
@click.argument("reason", required=False, default="switched off by hand")
def switch_off(reason: str) -> None:
    """Switch Cortex off in every session until ``cortex on``, saying why."""
    console.out(breaker.trip(reason).line())


@cortex.command()
def status() -> None:
    """Say whether Cortex is on, and if it's off, since when and why."""
    off = breaker.state()
    console.out("on" if off is None else off.line())


def _timestamp(session_id: str) -> str:
    """Mark this message's arrival and return the line saying what time it is.

    Every turn is logged, not because the line needs a record but because the corpus
    does: how many turns a day Alpha actually has is a number nobody has measured.
    """
    now = pendulum.now()
    since = timestamp_module.previous(session_id)
    timestamp_module.mark(session_id, now)
    log.write(
        "timestamp",
        session_id=session_id,
        gap_s=None if since is None else round(clock.gap_seconds(since, now), 1),
    )
    return timestamp_module.line(now, since)


def _round(value: float | None) -> float | None:
    """Round a sigma for the log, passing None through."""
    return None if value is None else round(value, 2)


if __name__ == "__main__":
    cortex()
