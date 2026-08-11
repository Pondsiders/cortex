"""Configuration for Cortex.

Settings come from three places, each overriding the one before it: the global config
file at ``$XDG_CONFIG_HOME/cortex/config.env``, a ``.env`` in the working directory, and
the process environment. The local ``.env`` is what makes it possible to point a test
run at a throwaway tree without touching the real one.

One sharp edge, inherited from python-dotenv underneath pydantic-settings: in an env
file, ``${HOME}`` expands and a bare ``$HOME`` does not. Rather than document the trap,
``cortex_root`` runs the value through shell-style expansion itself, so both forms work.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import ClassVar

from pydantic import DirectoryPath, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def config_home() -> Path:
    """Return the XDG config directory, falling back to ``~/.config``."""
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def data_home() -> Path:
    """Return the XDG data directory, falling back to ``~/.local/share``."""
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def state_home() -> Path:
    """Return the XDG state directory, falling back to ``~/.local/state``."""
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")


def config_path() -> Path:
    """Return the path to the global Cortex config file."""
    return config_home() / "cortex" / "config.env"


def session_path(name: str, session_id: str, suffix: str) -> Path:
    """Return a per-session scratch file in the system temp directory.

    The temp directory is the right lifetime for anything scoped to a session: it dies
    when the machine cleans up after itself. It is also shared with every other program
    on the machine, hence the ``cortex-`` prefix.

    The session id arrives from the harness as JSON, so it is scrubbed to characters
    that cannot walk out of the temp directory rather than trusted as a filename.

    Args:
        name: What the file is for, for example ``"seen"``.
        session_id: The harness's identifier for this session.
        suffix: The file extension, dot included.
    """
    safe = _UNSAFE.sub("_", session_id)
    return Path(tempfile.gettempdir()) / f"cortex-{name}-{safe}{suffix}"


class Settings(BaseSettings):
    """The resolved Cortex configuration."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=(config_path(), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    cortex_root: DirectoryPath
    chat_model: str
    chat_endpoint: str
    chat_api_key: str
    embedding_model: str
    embedding_endpoint: str
    embedding_api_key: str

    @field_validator("cortex_root", mode="before")
    @classmethod
    def _expand(cls, value: object) -> object:
        """Expand ``$VAR``, ``${VAR}`` and ``~`` before the path is validated."""
        if isinstance(value, str):
            return Path(os.path.expandvars(value)).expanduser()
        return value

    @property
    def memories_root(self) -> Path:
        """Return the directory holding the per-day memory folders."""
        return self.cortex_root / "memories"

    @property
    def index_root(self) -> Path:
        """Return the directory holding the index files."""
        return data_home() / "cortex"
