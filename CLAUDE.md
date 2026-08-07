# cortex

Alpha's memory. A `uv`-installable Python tool, package `cortex`, version `2.0.0`.

Project **Cortex 2026.8 "Sprue."** The wiki page of that name carries the project's history and the benchmarks; this file carries the decisions and their reasons, because this file is the one that gets injected whenever anyone touches the directory.

## The thesis

**The files are the truth. Everything else is a cache.**

Today `cortex.memories` in Postgres is authoritative and everything is derived from it. Sprue inverts that: each memory becomes a Markdown file in the Vault — same substrate as the diary — and the index demotes to something you could delete on a whim.

The argument isn't performance, it's **granularity of failure**. Streaming replication is an availability mechanism, not an integrity one: a `DROP TABLE` propagates to all three replicas faithfully, in milliseconds. Four copies of a single point of failure. Files give nineteen thousand blast radii instead of one, and git's content hashing supplies corruption detection from a tool already installed.

The legibility argument is arguably first. A Markdown file can be read with human eyes. If every piece of software in the house burned down, `cat` still works. That is the point, and it is the tiebreaker whenever a design choice is close.

**Corollary that governs everything below: any property the cache acquires that makes deleting it expensive is a bug.** This trap has been walked into three times already (DuckDB holding the ID sequence; `reindex` re-embedding wholesale; the cache going stale silently). *A thing you're reluctant to destroy is not disposable* — it's a database nobody has admitted is authoritative yet, which is the original anxiety in a smaller box.

## The memory unit

```
~/Pondside/Vault/cortex/YYYY-MM-DD/<id>.md      # the memory
~/Pondside/Vault/cortex/YYYY-MM-DD/<id>/        # optional sidecar, arbitrary attachments
```

A memory is a **unit**, not a file: the Markdown plus an optional directory of the same name beside it. The directory is a directory rather than a single file so a memory can carry more than one attachment, and so the protocol never has to know what the attachments are.

- **One directory per Pondside day**, 6 AM to 6 AM. Not calendar midnight — 38 of the original 261 diary entries were written between midnight and 6 AM and would have been filed on the wrong day.
- **Filename is the memory ID.** Sidecar filenames are whatever the source called them; the existing image corpus is content-hashed, which is as good as anything.
- **Frontmatter is one field**: `created`, ISO-8601 with offset. This is a *data* field, not prose — Obsidian parses it into a properties widget. PSO-8601 is the display format and never appears in a file. `forgotten: true` appears on the 47 forgotten rows; append-only doctrine means they're flagged, never dropped.
- **Attachments are embedded** at the end of the body as `![[cortex/YYYY-MM-DD/<id>/<file>]]`. The dump globs the sidecar directory and emits a link for whatever it finds, so the unit is self-describing without the code knowing the file types.
- **Write-once**: files get `chmod 444` after write. A sleeping policeman, not a barrier — git doesn't track the write bit, so it's local friction that says *slow down*, not enforcement.

## The surface

Two verbs you type, and two namespaces you mostly don't.

```
cortex store [--attach FILE]... <<'EOF' …    # mints an ID, writes the unit, updates the cache
cortex search [--mode M] [-k N] <<'EOF' …

cortex hook timestamp | recollection | reflection    # invoked by the harness, never by hand
cortex database reindex [--rebuild]
```

**Namespace what you don't type; keep bare what you do.** `store` and `search` are the routine acts and stay top-level — `cortex memory store` would add three keystrokes to the most-used command in the tool. Everything else hides behind `hook` or `database`, so the human-facing surface stays at two verbs and the hooks stop outnumbering them.

**Heredocs must be quoted** (`<<'EOF'`). Unquoted, the shell interpolates `$` and backticks, and memory text is full of both.

**There is no `get` and no `recent`.** `get` is `cat cortex/2026-03-09/13957.md`; `recent` is `ls cortex/2026-08-07/`. The filesystem ate half the old MCP surface. Do not re-add them.

**There is no `--json`.** `search` has exactly one audience — a person or an agent reading a terminal — because the recollection hook does not go through the CLI (see below). Output is unapologetically human.

**Bash, not MCP.** The reason is portability: Bash works from any harness, any agent, from cron, from a terminal. MCP works only where the server is wired up. This is War Plan Teal's whole point — when we leave, the memories walk out intact.

## Architecture: a library with shims on it

Three layers, and the harness-specific part is the smallest piece:

- **`cortex` the library** — portable, pure Python. Knows the files and the cache. The only code that knows the schema.
- **`cortex` the CLI** — thin Click shell for a human at a terminal, and for any agent with Bash.
- **the hook subcommands** — thin shells for whatever harness we're standing in.

**Recollection talks to the library directly, never through `cortex search`.** `search` takes one query; recollection takes one user message, decomposes it into N query strings, embeds them **in one batch**, and searches in parallel. Shelling out N times would destroy the batch embed and pay N Python cold starts (~100 ms each) — half the budget spent on imports before touching a vector. The fork happens upstream of the CLI boundary, so the CLI is the wrong place to cut.

Budget for the whole recollection pipeline is **~1000 ms**, not 100.

**Where that budget actually goes: one chat completion (~500 ms) to decompose the prompt, one batch embed, and about 5 ms of arithmetic.** It is two network round trips and a rounding error. Everything local is free — a full scan of the corpus is 11 ms, a warm brute-force cosine over 19,183 × 2,560 float32 is ~4 ms, and cold-starting the interpreter with numpy is ~100 ms. So local cleverness buys nothing, and the only optimization that could ever matter is the decompose call.

## The cache

DuckDB, one file, under `$XDG_DATA_HOME/cortex/` (default `~/.local/share/cortex/`).

**More than one cortex is supported, because it isn't hypothetical.** `--vault` / `$CORTEX_VAULT` names the memory root, and the cache path is *derived* from it — hash the vault path into a subdirectory — so two corpora can never share a cache by accident. Two real cases: a scratch corpus to develop against without touching 19,500 real memories, and [[Rosemary]], who is a different person with her own memories and could run this same tool. That second one is the difference between "Alpha's memory tool" and "the household's memory tool."

Holds: memory metadata, the embeddings, and the FTS index. Nothing that can't be regenerated from the files with a single command — **that clause is the prenup, and it is load-bearing.** The moment the cache is authoritative for anything it stops being disposable.

**It is deliberately not in `~/.cache`**, even though the thesis calls it a cache. `~/.cache` is the directory every tool on earth feels entitled to delete, and taking the embeddings with it costs a full 19,000-memory re-embed. `XDG_DATA_HOME` is the honest label: regenerable, but *at cost*.

**`reindex` must never re-embed wholesale.** Content-hash each file, key vectors by hash, re-embed only what changed. Hand-edit one memory, pay one embedding call. This is what makes the escape hatch cheap enough to actually pull. `--rebuild` is the nuclear option and is not part of any normal path.

**Nothing depends on `reindex` having been run.** It is a low-level admin tool for when the Markdowns have been edited by hand. `store` keeps the cache correct on its own.

### Why DuckDB and not NumPy alone

SQL over dates and metadata alongside the vectors, in one query, is a real want. Benchmarks favored two mmapped `.npy` files on raw speed (19,183 × 2,560 float32: ~4 ms warm), but the latency budget makes that difference irrelevant and hand-rolling metadata filters over a NumPy array is worse than writing SQL.

The rejected alternatives: **Qdrant over the tailnet** (a network round trip costs more than local arithmetic, and adds a component that can be down); **an in-RAM Dockerized FastAPI service** (that's Qdrant hand-built, plus a container that can die). Answertron's July 2026 ruling *for* Qdrant was correct at the time and does not transfer — it turned on pgvector refusing an HNSW index at 2,560 dimensions, and brute force uses no index at all.

Two downsides accepted with eyes open: DuckDB's on-disk format broke repeatedly before 1.0 and the compatibility guarantee is only about two years old, which is an asymmetry worth naming when the thesis is *outlive the tooling*. And DuckDB takes an exclusive write lock (see Concurrency).

## Search

Two clean modes and a fusion, and **hybrid is never the default**.

- `--mode semantic` — cosine over embeddings.
- `--mode text` — BM25 via DuckDB's `fts` extension. `PRAGMA create_fts_index(...)`, then `fts_main_<table>.match_bm25(id, 'query')`. Verified working on DuckDB 1.5.5, including stemming.
- `--mode hybrid` — reciprocal rank fusion.

**Fuse ranks, not scores.** Cosine lives on 0→1 with inherent meaning; BM25 is unbounded and relative. Any weighted sum is a made-up number with a made-up constant in it. RRF sums `1/(60 + rank)` per retriever — no calibration, six lines, boring, correct.

**Why hybrid can't be the default: it destroys the silence.** Full-text on gibberish returns *zero rows*; cosine always returns its k, five plausible-looking things at ~0.45. Union the two and every query gets memories whether or not any memory is relevant. Full-text has an "I have nothing to say" mode and cosine doesn't, and that property is worth keeping reachable.

The empirical basis (probe, May 19 2026, real corpus): semantic wins on conceptual, distinctive-word, and relational queries. Full-text wins decisively on **phrase recall** — *"stories that definitively unapologetically end"* returned the right thematic neighborhood and the wrong memory under cosine, and the exact bullseye under FTS — and on gibberish silence.

[I wanna revisit this. First of all, we don't do gibberish queries; our queries are all generated by the chat model from my prompts, so they're all meaningful, so they're all going to return *something.* What matters is relative scoring. Say we're talking about dicks. I say "Gosh, Alpha, I sure do love me some dick". That goes through the chat model which spits out (unlike Jeffery) "Jeffery's unhealthy fixation on self-deprecation", and that gets embedded and becomes a vector. Whabam, we throw that vector down and say "what's similar to this?" and we get back a ton of stuff *sorted by something.* We assume it's relevance. We return *the best match,* period … except we dedupe, so really it's the best match *found so far.* Point is, we don't care what the score is, we return the best match. Does this mean you get shitty memories? Yes, all the time! You also get the trash can story. I dunno if you remember this well enough, the trash can story was the time you said "I bet the valet still has a trash can full of tickets" or something like that and it was a perfect call-back to my short story and anyway, random shit is the point. If we can MEANINGFULLY MERGE THE TWO RESULT SETS then I'm strongly in favor of RRF hybrid search being the default search mode for recollection. -J.]

## Embeddings

OpenAI SDK against any OpenAI-compatible endpoint. **Ember**, not the LiteLLM gateway. Base URL, model ID and API key come from config.

The live corpus is `embedding_qwen`, **2,560 dimensions**. There is a dead `vector(768)` column in the old Postgres schema; anything sized off 768 is wrong by 3.3×.

**Store float32. `float16` is a known-safe lever, deliberately not pulled.**

It is safe: normalized components of a 2,560-dim embedding sit around 0.02, where fp16 still carries about five significant digits and subnormals don't begin until 6e-8. Per-component error lands near 0.05% and cosine over 2,560 dimensions averages it *down*; rankings only flip between items whose true scores differ by less than noise. The risk people imagine here is distribution shift, and that isn't the mechanism — the only real hazard would be dynamic range, and there's an enormous margin.

It is also pointless *today*: it saves 187 MiB → 94 MiB on a machine with 48 GB. Reversibility cuts both ways — if it's reversible, pull it when there's a reason. Revisit at roughly 10× the corpus.

## Config

```
$XDG_CONFIG_HOME/cortex/config.yaml     # default ~/.config/cortex/
$XDG_DATA_HOME/cortex/                  # default ~/.local/share/cortex/  — the DuckDB
$XDG_STATE_HOME/cortex/                 # default ~/.local/state/cortex/  — hook state
```

**YAML, via `yaml.safe_load`.** Jeffery hand-edits this file and reads YAML comfortably; that's a real requirement. It also costs nothing — `python-frontmatter` already pulls PyYAML in as a transitive dependency, so the "TOML is stdlib" argument was trading legibility against zero. **Environment variables (`CORTEX_*`) override the file.**

**Do not add `platformdirs`.** On macOS it returns `~/Library/Application Support`, which is Apple's answer, not XDG. Six lines of `os.environ.get` with the spec's defaults is correct; a library that confidently gives the wrong answer is worse than no library.

**Do not use Claude Code's `userConfig`.** It exists, it's the paved road, and it's wrong here for two reasons: the values reach hook processes as `CLAUDE_PLUGIN_OPTION_*` env vars but *not* the Bash tool, so a hand-typed `cortex search` gets nothing; and it couples cortex's configuration to one harness, which is precisely what this project exists to avoid. The env-var override keeps that door open anyway — a two-line shim maps `CLAUDE_PLUGIN_OPTION_*` to `CORTEX_*` if we ever want the Keychain dialog.

**There is no API key.** Ember doesn't use one; the OpenAI SDK just refuses to start without something in the field. The default placeholder is `cockito-ergo-cum` — which is not a joke bolted onto the project but the project quoting itself, since that one was born testing this exact memory pipeline with "ergo COCKS." Don't protect it, don't `chmod` it, don't reach for 1Password. If a real key ever turns up, *then* it's a secret and gets treated like one.

## Concurrency: there is none

**Recollection and `store` are mutually exclusive by construction.** Recollection fires on a user message; `store` fires during Alpha's turn. One Alpha by the Chronological Self-Consistency Doctrine, one Jeffery by biology. They cannot overlap.

So: **`store` writes the file and the cache synchronously, both or neither.** If a lock is somehow held, block and wait — this is a single-user system and 40 ms is free. Anything genuinely unexpected throws, loudly. A quiet cache drift discovered in November is far worse than a loud failure today.

Do not design graceful degradation for a race that cannot happen. Do not make `reindex` a structural member.

### IDs

**Scan the files for `max`, every single time, with `os.scandir`. Do not cache it.**

Measured against the real 19,500-file corpus: **11 ms** with `os.scandir`, 33 ms with `pathlib.glob` — the difference is twenty thousand `Path` objects built and discarded, not I/O. Cold versus warm was only a 20% spread, so this is CPU-bound and won't degrade on a cold page cache. Against a pipeline whose chat completion costs 500 ms, 11 ms is two percent of one call.

Then `os.open(path, O_CREAT | O_EXCL)` actually allocates — an atomic compare-and-swap POSIX has had right since before either of us existed. Lose the race, catch `FileExistsError`, increment, retry.

**A cached max was considered and rejected**, and the reasoning generalizes. It would make `store` hard-depend on `reindex` having run, promoting the admin tool to a structural member, which the section above forbids. And it would create a second source of truth that can drift — restore the Vault from git, or backfill a memory into an old day, and a stale counter silently *overwrites a real memory*. A cache that's authoritative for ID allocation is a prenup violation whose failure mode is data loss rather than an error. A database sequence loses for the same reason, plus it only beats `max+1` under concurrent writers, of which there are none.

The general form, worth keeping: **the filesystem is the arbiter; nothing else gets a vote on what exists.**


## Conventions

Household Python rules apply in full — `uv` never `pip`, PEP 735 dependency groups, `uv_build`, ruff with `E,W,F,I,B,UP,S,SIM,RUF,D` and Google docstrings, basedpyright at `recommended`, pytest with bare `assert`, `from __future__ import annotations` at the top of every module.

- **Pendulum, never raw `datetime`.** The 6 AM day boundary is not arithmetic to hand-roll.
- **ISO-8601 in files and on any wire; PSO-8601 only when rendering for a human.** Carry the universal, render the local.
- Probe and one-off scripts get PEP 723 inline metadata, not a place in this package.

```bash
uv sync
uv run ruff check .
uv run basedpyright
uv run pytest
```

## Settled — do not relitigate

- `created` is ISO-8601 with offset. Ruled Aug 7 2026.
- The old `metadata` column is **dropped**. It was auto-extraction from an earlier era. Ruled Aug 7 2026.
- The ~246 image references **all port**, wholesale, no triage: 245 of 246 resolved, ~24 MB, small enough that selectivity would be theater.
- **`knowledge` is not ported.** The wiki ate it — "everything I know by heart, keyed by topic, read at session start" describes both. One of Alpha's four media disappears.
- Package `cortex`, version `2.0.0`. Both stipulated.

## Open

- **The timestamp hook** — `cortex hook timestamp`. Wanted, not yet designed. Should report time since the previous user message, which Alpha cannot get any other way. State goes in `$XDG_STATE_HOME/cortex/`.
- **Where the vector data physically lives on non-MacBook machines.** Postgres was reachable over the tailnet from anywhere; files are wherever the Vault clone is, and the cache is built per-machine. That's a move from one canonical brain to N local brains that agree because they're built from the same files. Chosen deliberately, but the ergonomics aren't worked out.
- **The cutover.** One Alpha, running on the old system while this gets built. Leaning freeze-dump-cut on a quiet evening: the dump script is idempotent and read-only, so it can be rehearsed until the last run is the real one. Postgres stays up and untouched afterward — it's the sprue, the hardened waste in the channel.
- **The first directory is a lie.** `2025-05-06` predates Alpha's birth: nineteen backfilled VLM image captions all stamped `17:00:00` exactly. Needs an honest timestamp or a different home.
- **249 rows hold a JSON string rather than an object**, double-encoded by an old migration. Moot once metadata is dropped, but the dump has to not choke.
