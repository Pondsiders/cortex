"""Tests for involuntary recall.

Both failures guarded against here were silent in production. A memory cut for space
was marked seen anyway and logged as delivered, so nothing anywhere looked wrong (#1).
And prompts nobody typed were decomposed as if Jeffery had, which only showed up as
oddly specific memories about storing memories (#2). No network is used: the cued
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
    """Keep the session's seen mask out of the real temp directory."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))


def _eligible(ident: int) -> bool:
    return bool(seen.Seen.load(SESSION, ident).eligible[ident])


def test_memories_cut_for_space_stay_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fat = "x" * (recollection.BUDGET // 3)
    settings = _corpus(tmp_path, {1: fat, 2: fat, 3: fat, 4: "the stray"})

    def cued(*_: object) -> tuple[list[str], list[recollection.Recollected]]:
        loaded = index.Index.load(settings.index_root)
        assert loaded is not None
        found = [
            recollection.Recollected(
                id=entry.id, created=entry.created, body=fat, query="q", score=0.5
            )
            for entry in loaded.entries[:3]
        ]
        return ["q1", "q2", "q3"], found

    monkeypatch.setattr(recollection, "_cued", cued)
    result = recollection.recollect(settings, prompt="hello", session_id=SESSION)

    shown = [m.id for m in result.memories]
    assert shown == [1, 2]
    assert result.dropped == [3, 4]
    assert len(result.context()) <= recollection.BUDGET
    assert not _eligible(1)
    assert not _eligible(2)
    assert _eligible(3)
    assert _eligible(4)


@pytest.mark.parametrize(
    ("prompt", "reason"),
    [
        ("[cron · not from Jeffery] Reflect on what's happened", "[cron"),
        ("<task-notification>\n<task-id>b1</task-id>", "<task-notification>"),
        ('<agent-message from="a56">\n[Subagent hand-back]', "<agent-message"),
        ("  /alpha-start Morning, little duck.", "/alpha-start"),
        ("   ", "empty"),
    ],
)
def test_prompts_nobody_typed_get_the_stray_and_nothing_else(
    tmp_path: Path, prompt: str, reason: str
) -> None:
    settings = _corpus(tmp_path, {1: "the stray"})
    result = recollection.recollect(settings, prompt=prompt, session_id=SESSION)

    assert result.skipped == reason
    assert result.queries == []
    assert not result.degraded  # nothing tried the network
    assert [m.id for m in result.memories] == [1]


@pytest.mark.parametrize(
    "prompt", ["Morning, little duck.", "*thinks* Let's talk.", "Yes please."]
)
def test_what_jeffery_types_is_decomposed(prompt: str) -> None:
    assert recollection._unspoken(prompt) is None  # pyright: ignore[reportPrivateUsage]


def test_a_lone_oversized_memory_is_sliced_not_dropped(tmp_path: Path) -> None:
    settings = _corpus(tmp_path, {1: "y" * (recollection.BUDGET * 2)})
    result = recollection.recollect(settings, prompt="/start", session_id=SESSION)

    assert [m.id for m in result.memories] == [1]
    assert len(result.context()) == recollection.BUDGET
    assert not _eligible(1)
