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
from pathlib import Path
from typing import ClassVar

from pydantic import DirectoryPath, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
