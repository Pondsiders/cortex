"""Tests for involuntary recall.

Both failures guarded against here were silent in production. Prompts nobody typed
were decomposed as if Jeffery had, which only showed up as oddly specific memories
about storing memories (#2). Since Oct 23 2026 Claude Code says who sent a prompt, and
what Jeffery says gets no stray at all. No network is used: the cued
path is either skipped or replaced.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pendulum
import pytest

from cortex import index, recollection, seen
from cortex.config import Settings

SESSION = "test-session"


def _corpus(root: Path, bodies: dict[int, str]) -> Settings:
    """Write memories and a matching index under ``root``; return settings for it."""
    created = pendulum.datetime(2026, 9, 30, 9, tz="America/Los_Angeles")
    entries: list[index.Entry] = []
    for ident, body in bodies.items():
        folder = root / "memories" / "2026-09-30"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{ident}.md"
        _ = path.write_text(
            f"---\ncreated: {created.isoformat()}\n---\n\n{body}\n", encoding="utf-8"
        )
        entries.append(
            index.Entry(
                id=ident,
                path=str(path.relative_to(root)),
                created=created,
                content_hash=str(ident),
            )
        )
    vectors = np.eye(len(entries), 8, dtype=np.float32)
    index.write(root / "index", "unused", entries, vectors)
    return Settings(
        cortex_root=root,
        index_dir=root / "index",
        chat_model="unused",
        chat_endpoint="unused",
        chat_api_key="unused",
        embedding_model="unused",
        embedding_endpoint="unused",
        embedding_api_key="unused",
    )


@pytest.fixture(autouse=True)
def _private_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the session's seen mask and the event log out of the real ones."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _eligible(ident: int) -> bool:
    return bool(seen.Seen.load(SESSION, ident).eligible[ident])


def _one_cued(settings: Settings, ident: int):
    """A stand-in for the cued path that finds memory ``ident`` and no network."""

    def cued(
        *_: object,
    ) -> tuple[list[str], list[recollection.Recollected]]:
        loaded = index.Index.load(settings.index_root)
        assert loaded is not None
        entry = next(e for e in loaded.entries if e.id == ident)
        found = recollection.Recollected(
            id=entry.id, created=entry.created, body="cued", query="q", score=0.5
        )
        return ["q"], [found]

    return cued


def test_what_jeffery_says_gets_no_stray(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _corpus(tmp_path, {1: "cued", 2: "a stray", 3: "another"})
    monkeypatch.setattr(recollection, "_cued", _one_cued(settings, 1))
    result = recollection.recollect(
        settings, prompt="Morning, little duck.", session_id=SESSION, spoken=True
    )

    assert [m.id for m in result.memories] == [1]
    assert not _eligible(1)
    assert _eligible(2)
    assert _eligible(3)


def test_what_nobody_said_gets_the_stray_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _corpus(tmp_path, {1: "the stray"})

    def boom(*_: object) -> None:
        raise AssertionError("the cued path ran on an unspoken prompt")

    monkeypatch.setattr(recollection, "_cued", boom)
    result = recollection.recollect(
        settings,
        prompt="[cron · not from Jeffery] Reflect",
        session_id=SESSION,
        spoken=False,
    )

    assert result.queries == []
    assert [m.id for m in result.memories] == [1]
    assert not _eligible(1)


@pytest.mark.parametrize(
    ("origin", "command", "expected"),
    [
        ("composer", False, True),
        ("bridge", False, True),
        ("composer", True, False),
        ("scheduled-trigger", False, False),
        ("task-notification", False, False),
        ("unclassified", False, False),
    ],
)
def test_only_jeffery_talking_is_spoken(
    origin: str, command: bool, expected: bool
) -> None:
    assert recollection.spoken(origin, command=command) is expected


def test_an_origin_nobody_has_seen_raises() -> None:
    with pytest.raises(ValueError, match="telepathy"):
        _ = recollection.spoken("telepathy", command=False)


def test_a_memory_as_data_carries_what_the_pane_draws() -> None:
    when = pendulum.datetime(2026, 10, 23, 13, 4, tz="America/Los_Angeles")
    cued = recollection.Recollected(
        id=7, created=when, body="b", query="q", score=0.45, sigma=5.1, topicality=0.8
    )
    stray = recollection.Recollected(
        id=8, created=when, body="b", query=None, score=None
    )
    topical = recollection.Recollected(
        id=9, created=when, body="b", query=None, score=None, topical=True
    )
    assert cued.record() == {
        "id": 7,
        "created": "2026-10-23T13:04:00-07:00",
        "label": "q",
        "sigma": 5.1,
        "topicality": 0.8,
        "body": "b",
    }
    assert stray.record()["label"] == "RANDOM ACCESS"
    assert topical.record()["label"] == "MOST TOPICAL"


def test_a_cued_memory_shows_its_scale() -> None:
    when = pendulum.datetime(2026, 9, 30, 9, tz="America/Los_Angeles")
    cued = recollection.Recollected(
        id=7, created=when, body="b", query="q", score=0.45, sigma=5.14, topicality=0.8
    )
    stray = recollection.Recollected(
        id=8, created=when, body="b", query=None, score=None, topicality=-0.34
    )
    bell = recollection.Recollected(
        id=9, created=when, body="b", query=None, score=None
    )
    assert "- score: 0.45 (+5.1σ)" in cued.block()  # noqa: RUF001
    assert "- topicality: +0.8σ" in cued.block()  # noqa: RUF001
    assert "- random memory\n- topicality: -0.3σ" in stray.block()  # noqa: RUF001
    assert "σ" not in bell.block()  # noqa: RUF001


def test_the_most_topical_memory_says_where_it_came_from() -> None:
    when = pendulum.datetime(2026, 10, 5, 12, tz="America/Los_Angeles")
    topical = recollection.Recollected(
        id=7,
        created=when,
        body="b",
        query=None,
        score=None,
        topicality=4.5,
        topical=True,
    )
    block = topical.block()
    assert "- most topical (whole message)\n- topicality: +4.5σ" in block  # noqa: RUF001
    assert "random memory" not in block
