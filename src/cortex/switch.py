"""Recollection's on/off switch: one file, shared by every session.

The file being there means recollection is off. Nothing else reads or writes it, and
nothing switches it but a person.
"""

from __future__ import annotations

from pathlib import Path

from cortex.config import state_home


def path() -> Path:
    """Return the file whose presence means recollection is off."""
    return state_home() / "cortex" / "recollection-off"


def is_off() -> bool:
    """Say whether recollection is off."""
    return path().is_file()


def off() -> None:
    """Switch recollection off."""
    path().parent.mkdir(parents=True, exist_ok=True)
    path().touch()


def on() -> None:
    """Switch recollection on."""
    path().unlink(missing_ok=True)
