#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "openai", "python-frontmatter"]
# ///
"""Check that row *i* of vectors.npy really is the memory index.jsonl says it is.

A reindex proves the content hashes line up. It cannot prove the *rows* line up — an
off-by-one between the matrix and the entry list is silent, survives every hash check,
and makes recall return a plausible wrong memory forever. This re-embeds a sample and
compares each vector against the row the index claims for it.

Expect cosines around 0.999; the embedder is not bit-reproducible, so exact 1.0 is not
the bar. Anything near the corpus baseline of ~0.30 means the rows are shifted.

Usage:
    ./scripts/verify_alignment.py [--sample N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import frontmatter
import numpy as np
from openai import OpenAI

INDEX_ROOT = Path.home() / ".local/share/cortex"
BASELINE = 0.30
PASS_AT = 0.99


def main() -> int:
    """Sample the index, re-embed, and compare against the claimed rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--sample", type=int, default=12)
    _ = parser.add_argument("--root", type=Path, default=INDEX_ROOT)
    args = parser.parse_args()

    root: Path = args.root
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = [
        json.loads(line)
        for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    vectors = np.load(root / "vectors.npy", mmap_mode="r")

    print(f"manifest: {manifest['rows']:,} rows, {manifest['dimensions']}d, "
          f"{manifest['model']}")
    if not (len(entries) == vectors.shape[0] == manifest["rows"]):
        print("row counts disagree; reindex", file=sys.stderr)
        return 1

    cortex_root = Path(os.environ["CORTEX_ROOT"]).expanduser()
    client = OpenAI(
        base_url=os.environ["EMBEDDING_ENDPOINT"],
        api_key=os.environ["EMBEDDING_API_KEY"],
    )
    model = os.environ["EMBEDDING_MODEL"]

    rng = random.Random(19580)  # noqa: S311
    picks = sorted(rng.sample(range(len(entries)), min(args.sample, len(entries))))
    # Always include the ends, where an off-by-one shows up most starkly.
    picks = sorted({0, len(entries) - 1, *picks})

    print(f"\n{'row':>7}  {'memory':>8}  {'cosine':>9}   path")
    worst = 1.0
    for row in picks:
        entry = entries[row]
        body = frontmatter.loads(
            (cortex_root / entry["path"]).read_text(encoding="utf-8")
        ).content
        fresh = np.asarray(
            client.embeddings.create(model=model, input=[body]).data[0].embedding,
            dtype=np.float64,
        )
        stored = np.asarray(vectors[row], dtype=np.float64)
        cosine = float(
            stored @ fresh / (np.linalg.norm(stored) * np.linalg.norm(fresh))
        )
        worst = min(worst, cosine)
        flag = "" if cosine >= PASS_AT else "   <-- MISALIGNED"
        print(f"{row:>7}  {entry['id']:>8}  {cosine:>9.6f}   {entry['path']}{flag}")

    print(f"\nworst cosine over {len(picks)} rows: {worst:.6f}")
    if worst >= PASS_AT:
        print("PASS — rows are aligned")
        return 0
    print(f"FAIL — at least one row is wrong (baseline for unrelated is ~{BASELINE})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
