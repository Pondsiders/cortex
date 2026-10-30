"""Tests for the parts of Cortex whose failures are silent.

One test per way things have gone, or could go, quietly wrong: a result that depends on
the order the chat model happened to list its queries, a seen list that suppresses the
wrong memories, a memory filed under the wrong day, and a sync that re-embeds (or loses)
vectors it should have kept. None of them touches the network.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pendulum
import pytest
from openai import APIConnectionError

from cortex import (
    clock,
    embeddings,
    index,
    memories,
    pictures,
    recollection,
    seen,
    sync,
)
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


def test_sync_reads_only_the_files_that_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    # An index from before sizes and times were recorded: every entry lacks them.
    index.write(
        settings.index_root,
        "unused",
        [
            index.Entry(
                id=m.id, path=m.path, created=m.created, content_hash=m.content_hash
            )
            for m in memories.discover(settings.memories_root, tmp_path)
        ],
        np.eye(3, 4, dtype=np.float32),
    )

    reads: list[str] = []
    real_read = memories.read

    def counting_read(path: Path, root: Path) -> memories.Memory:
        reads.append(path.name)
        return real_read(path, root)

    monkeypatch.setattr(memories, "read", counting_read)

    # First sync after the upgrade reads everything once, embeds nothing, and records
    # the sizes and times.
    first = sync.sync(settings)
    assert sorted(reads) == ["1.md", "2.md", "3.md"]
    assert first.embedded == 0

    reads.clear()
    second = sync.sync(settings)
    assert reads == []
    assert (second.embedded, second.reused) == (0, 3)

    _ = (folder / "3.md").write_text(
        f"---\ncreated: {created}\n---\n\nmemory 3, revised\n", encoding="utf-8"
    )

    def fake_embed(
        _self: object, texts: list[str]
    ) -> Iterator[embeddings.EmbeddedBatch]:
        yield embeddings.EmbeddedBatch(
            indices=list(range(len(texts))), vectors=[[0.0, 0.0, 0.0, 1.0]] * len(texts)
        )

    monkeypatch.setattr(embeddings.Embedder, "embed_documents", fake_embed)
    reads.clear()
    third = sync.sync(settings)
    assert reads == ["3.md"]
    assert (third.embedded, third.reused) == (1, 2)


@pytest.mark.parametrize("damage", ["inconsistent", "other_model"])
def test_reindex_force_recovers_an_index_that_refuses_to_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
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
        embedding_model="new-model",
        embedding_endpoint="unused",
        embedding_api_key="unused",
    )
    on_disk = list(memories.discover(settings.memories_root, tmp_path))
    index.write(
        settings.index_root,
        "new-model" if damage == "inconsistent" else "old-model",
        [
            index.Entry(
                id=m.id, path=m.path, created=m.created, content_hash=m.content_hash
            )
            for m in on_disk
        ],
        np.eye(3, 4, dtype=np.float32),
    )
    if damage == "inconsistent":
        manifest = settings.index_root / index.MANIFEST
        _ = manifest.write_text(
            manifest.read_text(encoding="utf-8").replace('"rows": 3', '"rows": 4'),
            encoding="utf-8",
        )

    with pytest.raises(index.IndexError_, match="reindex --force"):
        _ = sync.sync(settings)

    def fake_embed(
        _self: object, texts: list[str]
    ) -> Iterator[embeddings.EmbeddedBatch]:
        yield embeddings.EmbeddedBatch(
            indices=list(range(len(texts))), vectors=[[0.0, 1.0, 0.0]] * len(texts)
        )

    def fake_dimensions(_self: object) -> int:
        return 3

    monkeypatch.setattr(embeddings.Embedder, "embed_documents", fake_embed)
    monkeypatch.setattr(embeddings.Embedder, "dimensions", fake_dimensions)

    result = sync.sync(settings, force=True)

    assert (result.embedded, result.reused) == (3, 0)
    rebuilt = index.Index.load(settings.index_root, expect_model="new-model")
    assert rebuilt is not None
    assert [e.id for e in rebuilt.entries] == [1, 2, 3]
    assert rebuilt.vectors.shape == (3, 3)


def test_a_sync_that_runs_out_of_patience_is_turned_away_while_another_holds_the_index(
    tmp_path: Path,
) -> None:
    (tmp_path / "memories").mkdir()
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

    with index.locked(settings.index_root, timeout=0), pytest.raises(index.Busy):
        _ = sync.sync(settings, timeout=0.1)

    # And the lock is released afterwards, by both of them.
    with index.locked(settings.index_root, timeout=0):
        pass


def test_writing_the_index_leaves_no_staging_files_behind(tmp_path: Path) -> None:
    created = pendulum.datetime(2026, 9, 30, 9, tz=LA)
    entries = [index.Entry(id=1, path="m/1.md", created=created, content_hash="h")]

    index.write(tmp_path, "unused", entries, np.eye(1, 4, dtype=np.float32))

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        index.ENTRIES,
        index.MANIFEST,
        index.VECTORS,
    ]


def test_a_memory_keeps_a_picture_and_sync_puts_it_on_the_shelf_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "memory"
    day = root / "memories" / "2026-10-30"
    day.mkdir(parents=True)
    settings = Settings(
        cortex_root=root,
        index_dir=tmp_path / "index",
        chat_model="unused",
        chat_endpoint="unused",
        chat_api_key="unused",
        embedding_model="unused",
        embedding_endpoint="unused",
        embedding_api_key="unused",
    )
    source = tmp_path / "Break_time.PNG"
    source.write_bytes(b"pretend this is a picture")

    name = pictures.keep(settings, source)
    assert pictures.keep(settings, source) == name
    assert name.endswith(".png")
    assert [p.name for p in settings.images_root.iterdir()] == [name]

    for ident in (1, 2):
        (day / f"{ident}.md").write_text(
            "---\ncreated: 2026-10-30T10:00:00-07:00\n"
            f"images:\n  - {name}\n---\n\nbody {ident}\n",
            encoding="utf-8",
        )

    def fake_text(
        _self: object, texts: list[str]
    ) -> Iterator[embeddings.EmbeddedBatch]:
        yield embeddings.EmbeddedBatch(
            indices=list(range(len(texts))), vectors=[[1.0, 0.0]] * len(texts)
        )

    monkeypatch.setattr(embeddings.Embedder, "embed_documents", fake_text)
    monkeypatch.setattr(embeddings.Embedder, "dimensions", lambda _self: 2)
    asked: list[int] = []

    def fake_eyes(_settings: Settings, encoded: list[str], **_k: object) -> np.ndarray:
        asked.append(len(encoded))
        return np.asarray([[1.0, 0.0]] * len(encoded), dtype=np.float32)

    monkeypatch.setattr(pictures, "embed", fake_eyes)

    assert sync.sync(settings).pictures == 1
    assert asked == [1]
    assert sync.sync(settings).pictures_embedded == 0
    assert asked == [1]

    (day / "2.md").write_text(
        "---\ncreated: 2026-10-30T10:00:00-07:00\nforgotten: true\n"
        f"images:\n  - {name}\n---\n\nbody 2\n",
        encoding="utf-8",
    )
    assert sync.sync(settings).pictures == 1
    (day / "1.md").unlink()
    (day / "3.md").write_text(
        "---\ncreated: 2026-10-30T10:00:00-07:00\n---\n\nno pictures\n",
        encoding="utf-8",
    )
    assert sync.sync(settings).pictures == 0


def test_goldeneye_away_leaves_the_memory_safe_and_the_next_sync_asks_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    source = tmp_path / "a.png"
    source.write_bytes(b"x")
    name = pictures.keep(settings, source)

    def gone(*_a: object, **_k: object) -> np.ndarray:
        raise APIConnectionError(request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(pictures, "embed", gone)
    result = pictures.sync(settings, [name], force=False)
    assert result.total == 0
    assert result.missed is not None

    monkeypatch.setattr(
        pictures, "embed", lambda *_a, **_k: np.eye(1, 4, dtype=np.float32)
    )
    assert pictures.sync(settings, [name], force=False).embedded == 1
