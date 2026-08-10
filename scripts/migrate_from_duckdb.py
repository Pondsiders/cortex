#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb", "numpy", "pendulum", "pytz"]
# ///
"""One-time: carry the vectors out of the retired DuckDB index into the file index.

Re-embedding all nineteen thousand memories takes about forty-three minutes of GPU. The
vectors already in DuckDB were written by a single code path and are perfectly good, so
this moves them across rather than making them again. It writes the same three files
``cortex reindex`` writes, carrying the content hashes over unchanged — which
is the point, because the proof that this worked is that the very next reindex finds
nothing stale.

Read-only against the DuckDB file. Run with --dry-run first.

Usage:
    ./scripts/migrate_from_duckdb.py [--dry-run] [--source PATH] [--dest PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pendulum

DEFAULT_SOURCE = Path.home() / ".local/share/cortex/database.duckdb"
DEFAULT_DEST = Path.home() / ".local/share/cortex"


def snapshot(source: Path) -> Path:
    """Clone the database so an open CLI session can't lock us out.

    APFS clones are copy-on-write, so this is effectively free; the plain copy is
    the fallback on filesystems that don't support it.
    """
    handle = Path(tempfile.mkdtemp()) / "snapshot.duckdb"
    if subprocess.run(  # noqa: S603
        ["/bin/cp", "-c", str(source), str(handle)], check=False
    ).returncode:
        _ = subprocess.run(["/bin/cp", str(source), str(handle)], check=True)  # noqa: S603
    return handle


def main() -> int:
    """Read the DuckDB index and write the file index beside it."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    _ = parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    _ = parser.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL"))
    _ = parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.model:
        print("EMBEDDING_MODEL is not set and --model was not given", file=sys.stderr)
        return 2
    if not args.source.is_file():
        print(f"no such database: {args.source}", file=sys.stderr)
        return 2

    con = duckdb.connect(str(snapshot(args.source)), read_only=True)
    rows = con.execute(
        "SELECT id, path, created, content_hash, embedding FROM memories ORDER BY id"
    ).fetchall()
    con.close()

    vectors = np.ascontiguousarray([row[4] for row in rows], dtype=np.float32)
    entries = [
        {
            "id": int(row[0]),
            "path": str(row[1]),
            "created": pendulum.instance(row[2]).isoformat(),
            "content_hash": str(row[3]),
        }
        for row in rows
    ]

    norms = np.linalg.norm(vectors, axis=1)
    print(f"rows            {len(rows):,}")
    print(f"vectors         {vectors.shape}  {vectors.nbytes / 1e6:.0f} MB")
    print(f"vector norms    min {norms.min():.5f}  max {norms.max():.5f}")
    print(f"ids             {entries[0]['id']} … {entries[-1]['id']}")
    print(f"model           {args.model}")

    if len(set(e["id"] for e in entries)) != len(entries):
        print("duplicate ids in the source; refusing to write", file=sys.stderr)
        return 1
    if not np.isfinite(vectors).all():
        print("non-finite values in the vectors; refusing to write", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    with (dest / "vectors.tmp.npy").open("wb") as handle:
        np.save(handle, vectors)
        handle.flush()
        os.fsync(handle.fileno())
    with (dest / "index.jsonl.tmp").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    with (dest / "manifest.json.tmp").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "dimensions": int(vectors.shape[1]),
                "rows": len(entries),
                "written": pendulum.now().isoformat(),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(dest / "vectors.tmp.npy", dest / "vectors.npy")
    os.replace(dest / "index.jsonl.tmp", dest / "index.jsonl")
    os.replace(dest / "manifest.json.tmp", dest / "manifest.json")
    print(f"\nwrote three files to {dest}")
    print("next: cortex reindex  (it should report nothing stale)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
