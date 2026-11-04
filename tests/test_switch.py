"""Tests for recollection's on/off switch.

Off means off: the hook adds nothing to the prompt at all, not even the time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cortex import switch
from cortex.cli import cortex


@pytest.fixture(autouse=True)
def _private_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the switch out of the real one."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def test_on_and_off_from_the_command_line() -> None:
    runner = CliRunner()
    assert runner.invoke(cortex, ["recollection", "status"]).output == "on\n"
    assert runner.invoke(cortex, ["recollection", "off"]).output == "off\n"
    assert switch.is_off()
    assert runner.invoke(cortex, ["recollection", "status"]).output == "off\n"
    assert runner.invoke(cortex, ["recollection", "on"]).output == "on\n"
    assert not switch.is_off()


def test_while_off_a_prompt_gets_nothing() -> None:
    switch.off()
    packet = {
        "text": "Morning, little duck.",
        "origin": "composer",
        "command": None,
        "session_id": "test-session",
        "attachments": [],
    }
    result = CliRunner().invoke(cortex, ["hook", "prompt"], input=json.dumps(packet))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"context": [], "memories": [], "off": True}
