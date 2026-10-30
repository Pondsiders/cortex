"""The pictures: the files I keep, and the shelf of vectors that finds them.

A picture joins a memory on purpose, through ``cortex store --image``. Its bytes go in
``$CORTEX_ROOT/images/<sha256>.<ext>``, so storing the same bytes twice writes one file,
and the memory names that file in its frontmatter. The file and the frontmatter are
authoritative. Everything below them is derived.

The shelf is one ``pictures.npz`` beside the text index, holding one row per picture
rather than per memory: names, vectors, and the model that made them. It's one file
replaced atomically, so unlike the text index it can't disagree with itself. Only
pictures some live memory names get a row, so forgetting a picture's last memory takes
it off the shelf.

Goldeneye makes the vectors (``Infra/gargamel``). These vectors only ever meet each
other: they share no space with Qwen's, and a picture reaches text only through the
memory it hangs from. Nothing here decides what counts as the same picture. A hit comes
back with its cosine and z-score, and judging them is the reader's job.
"""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import final

import numpy as np
from openai import APIError, OpenAI

from cortex.config import Settings

SHELF = "pictures.npz"
BATCH_SIZE = 16
SUFFIXES = frozenset({".gif", ".jpeg", ".jpg", ".png", ".webp"})
"""What Goldeneye's Pillow reads and Claude's Read tool shows."""


class PictureError(Exception):
    """Raised when a picture can't be kept, or one a memory names has gone."""


def keep(settings: Settings, source: Path) -> str:
    """Copy a picture into the images folder, named by its hash.

    Bytes already kept are left alone, which is all deduplication is.

    Args:
        settings: Says where the images folder is.
        source: The picture to keep.

    Returns:
        The kept file's name, for the memory's frontmatter.

    Raises:
        PictureError: If the file isn't a kind of picture we can use.
    """
    suffix = source.suffix.lower()
    if suffix not in SUFFIXES:
        msg = f"{source}: expected one of {sorted(SUFFIXES)}"
        raise PictureError(msg)
    data = source.read_bytes()
    name = f"{hashlib.sha256(data).hexdigest()}{suffix}"
    settings.images_root.mkdir(parents=True, exist_ok=True)
    target = settings.images_root / name
    if not target.exists():
        _replace(target, data)
    return name


def _replace(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` atomically."""
    fd, staged = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staged, 0o644)
        os.replace(staged, target)
    finally:
        Path(staged).unlink(missing_ok=True)


def embed(
    settings: Settings, encoded: Sequence[str], *, timeout: float, max_retries: int
) -> np.ndarray:
    """Ask Goldeneye for unit vectors, one per base64 picture, in the order given.

    Raises:
        openai.APIError: If Goldeneye is missing or refuses.
    """
    client = OpenAI(
        base_url=settings.vision_endpoint or settings.embedding_endpoint,
        api_key=settings.embedding_api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    response = client.embeddings.create(
        model=settings.vision_model, input=list(encoded), encoding_format="float"
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = np.asarray([item.embedding for item in ordered], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@final
@dataclass(frozen=True)
class Shelf:
    """Every kept picture some live memory names, with its vector."""

    names: list[str]
    vectors: np.ndarray
    model: str

    @classmethod
    def load(cls, root: Path) -> Shelf | None:
        """Read the shelf in ``root``, or None if there isn't one yet."""
        path = root / SHELF
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            return cls(
                names=[str(name) for name in data["names"]],
                vectors=data["vectors"],
                model=str(data["model"]),
            )

    def save(self, root: Path) -> None:
        """Replace the shelf in ``root`` atomically."""
        root.mkdir(parents=True, exist_ok=True)
        fd, staged = tempfile.mkstemp(dir=root, prefix=f".{SHELF}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(
                    handle,
                    names=np.asarray(self.names, dtype=np.str_),
                    vectors=self.vectors,
                    model=np.asarray(self.model),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged, root / SHELF)
        finally:
            Path(staged).unlink(missing_ok=True)


@dataclass(frozen=True)
class Result:
    """What a shelf sync did."""

    total: int
    embedded: int
    missed: str | None
    """Why some pictures have no vector yet; None when every one has."""


def sync(settings: Settings, named: Iterable[str], *, force: bool) -> Result:
    """Bring the shelf in line with the pictures live memories name.

    Goldeneye being away is no emergency: the memories are safe, what could be done is
    kept, and whichever sync runs next asks again.

    Args:
        settings: Where the pictures and the shelf live, and which model to ask.
        named: Every picture name in a live memory's frontmatter.
        force: Re-embed everything.

    Returns:
        Counts, and why anything was left undone.

    Raises:
        PictureError: If a memory names a picture that isn't on disk.
    """
    wanted = sorted(set(named))
    old = None if force else Shelf.load(settings.index_root)
    if old is not None and old.model != settings.vision_model:
        old = None
    known = {} if old is None else dict(zip(old.names, old.vectors, strict=True))

    fresh = [name for name in wanted if name not in known]
    if not fresh and old is not None and old.names == wanted:
        return Result(total=len(wanted), embedded=0, missed=None)
    if not wanted and old is None and not (settings.index_root / SHELF).exists():
        return Result(total=0, embedded=0, missed=None)

    missed: str | None = None
    embedded = 0
    for start in range(0, len(fresh), BATCH_SIZE):
        batch = fresh[start : start + BATCH_SIZE]
        try:
            vectors = embed(
                settings,
                [_encoded(settings, name) for name in batch],
                timeout=120.0,
                max_retries=2,
            )
        except APIError as error:
            missed = f"{settings.vision_model}: {type(error).__name__}"
            break
        known.update(zip(batch, vectors, strict=True))
        embedded += len(batch)

    names = [name for name in wanted if name in known]
    matrix = (
        np.stack([known[name] for name in names])
        if names
        else np.zeros((0, 0), dtype=np.float32)
    )
    Shelf(names, matrix, settings.vision_model).save(settings.index_root)
    return Result(total=len(names), embedded=embedded, missed=missed)


def _encoded(settings: Settings, name: str) -> str:
    """Read a kept picture as base64, or say loudly that it's gone."""
    path = settings.images_root / name
    if not path.is_file():
        msg = f"a memory names {name}, which isn't in {settings.images_root}"
        raise PictureError(msg)
    return base64.b64encode(path.read_bytes()).decode("ascii")
