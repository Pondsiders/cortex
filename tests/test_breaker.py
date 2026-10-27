"""Tests for the circuit breaker.

The breaker exists so an outage costs one loud turn rather than a stack trace on every
prompt until somebody notices. The thing it must never do is hide a bug: only a model
that can't be reached trips it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx
import openai
import pytest
from click.testing import CliRunner

from cortex import breaker, recollection
from cortex.cli import cortex
from cortex.config import Settings


@pytest.fixture(autouse=True)
def _private_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the switch, the clock marks and the event log out of the real ones."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))


def _settings(root: Path) -> Settings:
    return Settings(
        cortex_root=root,
        chat_model="gemma",
        chat_endpoint="http://127.0.0.1:9/v1",
        chat_api_key="unused",
        embedding_model="qwen",
        embedding_endpoint="http://127.0.0.1:9/v1",
        embedding_api_key="unused",
    )


def test_off_and_on_again() -> None:
    assert breaker.state() is None
    off = breaker.trip("gargamel is busy")
    assert breaker.state() == off
    assert "(gargamel is busy)" in off.line()
    assert "`cortex on`" in off.line()
    breaker.reset()
    assert breaker.state() is None


def test_an_unreachable_model_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*_: object, **__: object) -> list[str]:
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(recollection.Decomposer, "decompose", refused)
    with pytest.raises(recollection.Unreachable) as caught:
        _ = recollection._cued(  # pyright: ignore[reportPrivateUsage]
            _settings(tmp_path),
            "hi",
            None,
            None,
            1.0,  # pyright: ignore[reportArgumentType]
        )
    assert caught.value.reason == "chat model gemma unreachable: APIConnectionError"


def test_a_model_that_answers_wrongly_is_a_bug_not_an_outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: object, **__: object) -> list[str]:
        raise openai.InternalServerError(
            "boom",
            response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
            body=None,
        )

    monkeypatch.setattr(recollection.Decomposer, "decompose", broken)
    with pytest.raises(openai.InternalServerError):
        _ = recollection._cued(  # pyright: ignore[reportPrivateUsage]
            _settings(tmp_path),
            "hi",
            None,
            None,
            1.0,  # pyright: ignore[reportArgumentType]
        )


def test_while_off_a_prompt_gets_the_time_and_the_reminder_and_nothing_else() -> None:
    _ = breaker.trip("switched off by hand")
    packet = {
        "text": "Morning, little duck.",
        "origin": "composer",
        "command": None,
        "session_id": "test-session",
        "attachments": [],
    }
    result = CliRunner().invoke(cortex, ["hook", "prompt"], input=json.dumps(packet))

    assert result.exit_code == 0, result.output
    answer = json.loads(result.output)
    assert len(answer["context"]) == 2
    assert answer["context"][0].startswith("cortex timestamp:")
    assert answer["context"][1].startswith("cortex: off since")
    assert answer["memories"] == []
    assert answer["off"]["reason"] == "switched off by hand"


def test_the_switch_from_the_command_line() -> None:
    runner = CliRunner()
    assert runner.invoke(cortex, ["status"]).output == "on\n"
    assert (
        "(MOSS has the GPU)"
        in runner.invoke(cortex, ["off", "MOSS has the GPU"]).output
    )
    assert runner.invoke(cortex, ["status"]).output.startswith("cortex: off since")
    status = json.loads(runner.invoke(cortex, ["status", "--json"]).output)
    assert status["reason"] == "MOSS has the GPU"
    assert runner.invoke(cortex, ["on"]).output == "on\n"
    assert breaker.state() is None
    assert runner.invoke(cortex, ["status", "--json"]).output == "null\n"
