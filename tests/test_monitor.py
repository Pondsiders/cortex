"""Tests for the quiet watcher.

Every failure guarded against here is a *silence*: a scan that stops noticing new
memories, a watcher that never speaks, a pause that never lifts. Brittle code
self-tests only where it fails loudly, and none of these do — each one looks exactly
like a watcher with nothing to report, which is the one thing it must never look like.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cortex import memories, monitor
from cortex.config import Settings

# Beyond any pid this platform will hand out, so nothing is ever listening on it.
DEAD_PID = 999999


def _settings(root: Path) -> Settings:
    (root / "memories").mkdir(exist_ok=True)
    return Settings(
        cortex_root=root,
        chat_model="unused",
        chat_endpoint="unused",
        chat_api_key="unused",
        embedding_model="unused",
        embedding_endpoint="unused",
        embedding_api_key="unused",
    )


def _memory(root: Path, day: str, ident: int, body: str) -> Path:
    folder = root / "memories" / day
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{ident}.md"
    _ = path.write_text(
        f"---\ncreated: 2026-08-21T09:00:00-07:00\n---\n\n{body}\n", encoding="utf-8"
    )
    return path


def test_latest_notices_a_day_folder_that_did_not_exist_before(tmp_path: Path) -> None:
    """No day is remembered between scans, so the 6 AM seam is a non-event."""
    root = tmp_path / "memories"
    root.mkdir()

    _ = _memory(tmp_path, "2026-08-20", 7, "yesterday")
    first = memories.latest(root)
    assert first is not None
    assert first.stem == "7"

    _ = _memory(tmp_path, "2026-08-21", 8, "today")
    second = memories.latest(root)
    assert second is not None
    assert second.stem == "8"


def test_watch_speaks_only_once_the_quiet_is_long_enough(tmp_path: Path) -> None:
    """And when it speaks it names the memory it is measuring from."""
    settings = _settings(tmp_path)
    _ = _memory(tmp_path, "2026-08-21", 20192, "**The outer product is the parent**")

    said: list[str] = []
    monitor.watch(settings, interval=3600, poll=0, say=said.append, forever=False)
    assert len(said) == 1
    assert said[0].startswith("cortex monitor armed")

    said.clear()
    monitor.watch(settings, interval=0, poll=0, say=said.append, forever=False)
    report = next(line for line in said if line.startswith("No memories stored"))
    assert "#20192" in report
    assert "The outer product is the parent" in report


def test_a_pause_lock_hushes_the_watcher_unless_its_owner_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock left behind by a dead process is debris, not an instruction."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    settings = _settings(tmp_path)
    _ = _memory(tmp_path, "2026-08-21", 1, "anything")

    monitor.hold()
    said: list[str] = []
    monitor.watch(settings, interval=0, poll=0, say=said.append, forever=False)
    assert any("paused" in line for line in said)
    assert not any(line.startswith("No memories stored") for line in said)
    assert monitor.holder() == os.getpid()

    _ = monitor.lock_path().write_text(f"{DEAD_PID}\n", encoding="utf-8")
    said.clear()
    monitor.watch(settings, interval=0, poll=0, say=said.append, forever=False)
    assert any(line.startswith("No memories stored") for line in said)
    assert not monitor.lock_path().exists()
