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

from cortex import clock, log
from cortex import recollection as recollection_module
from cortex import reflection as reflection_module
from cortex import search as search_module
from cortex import sync as sync_module
from cortex.config import Settings
from cortex.embeddings import BATCH_SIZE, CONCURRENCY


def _say(message: str) -> None:
    """Write a status line to stderr, safely alongside an active progress bar."""
    tqdm.write(message, file=click.get_text_stream("stderr"))


def _run_sync(
    settings: Settings, *, force: bool, batch_size: int, concurrency: int
) -> sync_module.Result:
    """Run a sync with a progress bar, returning its result."""
    bar: tqdm[Any] | None = None

    def on_start(count: int) -> None:
        nonlocal bar
        if count:
            _say(f"embedding {count} memories at {concurrency} x {batch_size}")
            bar = tqdm(total=count, unit="mem", smoothing=0.05)

    def on_progress(landed: int) -> None:
        if bar is not None:
            _ = bar.update(landed)

    try:
        return sync_module.sync(
            settings,
            force=force,
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
    click.echo(path.relative_to(settings.cortex_root))

    result = _run_sync(
        settings, force=False, batch_size=BATCH_SIZE, concurrency=CONCURRENCY
    )
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
    largest = 0
    for day in os.scandir(memories_root):
        if not day.is_dir():
            continue
        for entry in os.scandir(day.path):
            name = entry.name
            if name.endswith(".md") and name[:-3].isdigit():
                largest = max(largest, int(name[:-3]))
    return largest


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
    click.echo(f"{results.corpus:,} memories · this query's corpus baseline {scale}")
    for hit in results.hits:
        when = pendulum.instance(hit.created).format("ddd MMM D YYYY, h:mm A")
        # ruff reads the sigma as a confusable 'o'; it's display text, not a name.
        sigma = f"{hit.sigma:+.1f}σ"  # noqa: RUF001
        click.echo(f"\n#{hit.id}  {hit.score:.4f}  {sigma}  {when}")
        click.echo(f"{hit.path}\n")
        click.echo(hit.body)


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
    """Claude Code hook scripts. Each reads its event JSON from standard input."""


@hook.command("recollection")
def hook_recollection() -> None:
    """Recall memories for a UserPromptSubmit event and return them as context.

    Recollection is enrichment rather than a gate, so nothing here is allowed to cost
    Jeffery his turn. In particular it must never exit 2, which Claude Code reads as a
    blocking error on this event: it discards the prompt. Every failure exits 1 with a
    line on stderr, which the harness shows in the transcript and then carries on.
    """
    started = time.monotonic()
    try:
        event: dict[str, Any] = json.loads(sys.stdin.read())
        prompt = str(event.get("prompt", ""))
        session_id = str(event["session_id"])
    except (ValueError, KeyError) as error:
        raise SystemExit(_bail(f"unusable hook input: {error}")) from error

    try:
        settings = Settings()  # pyright: ignore[reportCallIssue]
        result = recollection_module.recollect(
            settings, prompt=prompt, session_id=session_id
        )
        context = result.context()
    # Every failure is caught, because a hook that raises is a hook that costs a turn.
    except Exception as error:
        raise SystemExit(_bail(f"recollection failed: {error}")) from error

    elapsed = int((time.monotonic() - started) * 1000)
    log.write(
        "recollection",
        session_id=session_id,
        prompt_id=event.get("prompt_id"),
        ms=elapsed,
        degraded=result.degraded,
        prompt=prompt,
        queries=[
            {"q": m.query, "id": m.id, "score": round(m.score, 4)}
            for m in result.memories
            if m.query is not None and m.score is not None
        ],
        unanswered=[
            q for q in result.queries if q not in {m.query for m in result.memories}
        ],
        lagniappe=next((m.id for m in result.memories if m.query is None), None),
        chars=len(context),
    )

    if not context:
        return
    click.echo(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                }
            }
        )
    )


@hook.command("reflection")
def hook_reflection() -> None:
    """Ask Claude to store a memory on a Stop event, every third turn.

    Like recollection this must never exit 2, which Claude Code reads as a blocking
    error. Unlike recollection, a Stop hook that keeps the conversation going is fired
    again once Claude has answered, so the second firing is dropped rather than counted.
    """
    try:
        event: dict[str, Any] = json.loads(sys.stdin.read())
        session_id = str(event["session_id"])
    except (ValueError, KeyError) as error:
        raise SystemExit(_bail(f"unusable hook input: {error}")) from error

    if event.get("stop_hook_active"):
        return

    try:
        turn = reflection_module.bump(session_id)
    # A hook that raises is a hook that costs a turn.
    except OSError as error:
        raise SystemExit(_bail(f"reflection failed: {error}")) from error

    if not reflection_module.due(turn):
        return

    log.write("reflection", session_id=session_id, turn=turn)
    click.echo(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "Stop",
                    "additionalContext": reflection_module.PROMPT,
                }
            }
        )
    )


def _bail(message: str) -> int:
    """Report a hook failure without blocking the turn, and return its exit code."""
    print(f"cortex: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    cortex()
