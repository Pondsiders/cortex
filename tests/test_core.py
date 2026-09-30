"""Tests for the parts of Cortex whose failures are silent.

One test per way things have gone, or could go, quietly wrong: a result that depends on
the order the chat model happened to list its queries, a seen list that suppresses the
wrong memories, a memory filed under the wrong day, and a sync that re-embeds (or loses)
vectors it should have kept. None of them touches the network.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pendulum
import pytest

from cortex import clock, index, memories, recollection, seen, sync
from cortex.config import Settings

LA = "America/Los_Angeles"


def test_contested_memory_goes_to_the_stronger_query_whatever_the_order() -> None:
    # Both queries want row 0; B wants it more. A's second choice is row 1.
    a = [0.6, 0.5, 0.1]
    b = [0.9, 0.2, 0.1]

    first = recollection._assign(np.array([a, b]), ["A", "B"])  # pyright: ignore[reportPrivateUsage]
    second = recollection._assign(np.array([b, a]), ["B", "A"])  # pyright: ignore[reportPrivateUsage]

    assert first == {0: (1, 0.5), 1: (0, 0.9)}
    assert second == {0: (0, 0.9), 1: (1, 0.5)}


def test_seen_list_is_keyed_by_id_and_new_memories_start_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    mask = seen.Seen.load("s", largest_id=10)
    mask.mark([3, 7])
    mask.save()

    # The corpus grows mid-session: 11 and 12 were stored after the session began.
    grown = seen.Seen.load("s", largest_id=12)
    ids = np.array([12, 7, 5, 11, 3])
    assert grown.filter(ids).tolist() == [False, False, True, False, False]


@pytest.mark.parametrize(
    ("hour", "minute", "day"),
    [(0, 30, "2026-09-29"), (5, 59, "2026-09-29"), (6, 0, "2026-09-30")],
)
def test_the_day_turns_at_six_in_the_morning(hour: int, minute: int, day: str) -> None:
    moment = pendulum.datetime(2026, 9, 30, hour, minute, tz=LA)
    assert clock.pondside_day(moment) == day


def test_sync_keeps_unchanged_vectors_and_drops_forgotten_memories(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "memories" / "2026-09-30"
    folder.mkdir(parents=True)
    created = pendulum.datetime(2026, 9, 30, 9, tz=LA).isoformat()
    for ident in (1, 2, 3):
        _ = (folder / f"{ident}.md").write_text(
            f"---\ncreated: {created}\n---\n\nmemory {ident}\n", encoding="utf-8"
        )

    settings = Settings(
        cortex_root=tmp_path,
        index_dir=tmp_path / "index",
        chat_model="unused",
        chat_endpoint="unused",
        chat_api_key="unused",
        embedding_model="unused",
        embedding_endpoint="unused",
        embedding_api_key="unused",
    )
    on_disk = list(memories.discover(settings.memories_root, tmp_path))
    vectors = np.eye(3, 4, dtype=np.float32)
    index.write(
        settings.index_root,
        "unused",
        [
            index.Entry(
                id=m.id, path=m.path, created=m.created, content_hash=m.content_hash
            )
            for m in on_disk
        ],
        vectors,
    )

    _ = (folder / "2.md").write_text(
        f"---\ncreated: {created}\nforgotten: true\n---\n\nmemory 2\n",
        encoding="utf-8",
    )
    result = sync.sync(settings)

    assert (result.embedded, result.reused, result.dropped) == (0, 2, 1)
    rebuilt = index.Index.load(settings.index_root)
    assert rebuilt is not None
    assert [e.id for e in rebuilt.entries] == [1, 3]
    assert np.array_equal(np.asarray(rebuilt.vectors), vectors[[0, 2]])
