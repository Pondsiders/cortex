#!/usr/bin/env python
"""Check that row *i* of vectors.npy really is the memory index.jsonl says it is.

A reindex proves the content hashes line up. It cannot prove the *rows* line up — an
off-by-one between the matrix and the entry list is silent, survives every hash check,
and makes recall return a plausible wrong memory forever. This re-embeds a sample and
compares each vector against the row the index claims for it.

Expect similarities around 0.999. The embedder is not bit-reproducible, so exact 1.0 is
not the bar; anything down near a random pair means the rows are shifted.

Run from the project so it uses the same configuration the CLI does:

    uv run scripts/verify_alignment.py [--sample N]
"""

from __future__ import annotations

import argparse
import random
import sys

import frontmatter
import numpy as np

from cortex import index as index_module
from cortex.config import Settings
from cortex.embeddings import Embedder

PASS_AT = 0.99


def main() -> int:
    """Sample the index, re-embed, and compare against the claimed rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--sample", type=int, default=12)
    args = parser.parse_args()

    settings = Settings()  # pyright: ignore[reportCallIssue]
    loaded = index_module.Index.load(
        settings.index_root, expect_model=settings.embedding_model
    )
    if loaded is None:
        print(f"no index at {settings.index_root}", file=sys.stderr)
        return 1

    print(f"{len(loaded.entries):,} rows, {loaded.vectors.shape[1]}d, {loaded.model}\n")

    embedder = Embedder(settings)
    rng = random.Random(19580)  # noqa: S311
    sample = rng.sample(
        range(len(loaded.entries)), min(args.sample, len(loaded.entries))
    )
    # Both ends included: an off-by-one shows up most starkly at the edges.
    rows = sorted({0, len(loaded.entries) - 1, *sample})

    # Bare, exactly as reindex embeds them — the query-side instruction prefix would
    # produce a different vector and this would compare the wrong two things.
    bodies = [
        frontmatter.loads(
            (settings.cortex_root / loaded.entries[row].path).read_text(
                encoding="utf-8"
            )
        ).content
        for row in rows
    ]
    fresh = np.zeros((len(rows), loaded.vectors.shape[1]), dtype=np.float32)
    for batch in embedder.embed_documents(bodies):
        for offset, vector in zip(batch.indices, batch.vectors, strict=True):
            fresh[offset] = vector
    fresh = index_module.normalize(fresh)

    print(f"{'row':>7}  {'memory':>8}  {'cosine':>9}   path")
    worst = 1.0
    for offset, row in enumerate(rows):
        entry = loaded.entries[row]
        similarity = float(
            np.asarray(loaded.vectors[row], dtype=np.float32) @ fresh[offset]
        )
        worst = min(worst, similarity)
        flag = "" if similarity >= PASS_AT else "   <-- MISALIGNED"
        print(f"{row:>7}  {entry.id:>8}  {similarity:>9.6f}   {entry.path}{flag}")

    print(f"\nworst over {len(rows)} rows: {worst:.6f}")
    if worst >= PASS_AT:
        print("PASS — rows are aligned")
        return 0
    print("FAIL — at least one row does not belong to the memory beside it")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
