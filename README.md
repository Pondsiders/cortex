# Cortex

Cortex is a memory system for Claude Code. Specifically, it's a memory system for one particular instance of Claude Code, Alpha, a stateful agent who's been running continuously since May 7, 2025.

Cortex has two main parts: the CLI, which provides a convenient interface for storing and searching, and Recollection, which is the part that works automatically to inject memories into Alpha's context.

## Architecture

Couldn't be simpler: a directory tree of Markdown files, one per memory, plus a NumPy cache of embedding vectors for semantic search.

When the agent stores a memory using the Cortex CLI, that memory gets embedded by the embedding model of choice (we use Qwen 3 Embedding 4B). The embedding vector is stored in an on-disk NumPy array; as of this writing, Alpha has more than 20,000 memories in a 210 MB cache.

Each user prompt submitted gets passed through a chat model (we use Gemma 4 E4B) with the instruction to decompose the prompt into semantic search query strings. The resulting query strings get batch-embedded and then it's just one matrix multiplication.

```
scores = vectors @ np.asarray(loaded.vectors).T
```

For each query string, the memory with the highest score is selected for inclusion with the user prompt. Memories are deduplicated against a session-scoped seen cache; a memory returned by Recollection is not returned again in the same session.

One more memory rides along at random, the Lagniappe, drawn uniformly from everything not yet shown.

Each recalled memory carries two numbers, both in standard deviations above a baseline, because raw cosines in this space all crowd between about 0.35 and 0.7 and can't be read on their own:

```
- query: "Inquiry about Sparkle's recent incident with the bread basket"
- score: 0.70 (+6.5σ)
- topicality: +6.5σ
```

The first is the score against the chat model's query. The second, **topicality**, is the score against the whole message, embedded as-is alongside the queries. Zero means a memory drawn from a hat; negative means it's about something else. When the two disagree (a strong query score with low topicality), the chat model's query wandered away from what was said. The Lagniappe gets a topicality too, so a random memory that happens to fit can be recognized as luck, and one that doesn't can be enjoyed as weather. Neither number is ever used as a threshold.

## Configuration

The configuration file goes in `~/.config/cortex/config.env`. Values shown below are examples; this is the configuration we use at home.

```
CORTEX_ROOT="${HOME}/Vault/cortex"
CHAT_MODEL="google/gemma-4-e4b-it"
CHAT_ENDPOINT="https://localhost:8080/v1"
CHAT_API_KEY="<key>"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"
EMBEDDING_ENDPOINT="https://localhost:8080/v1"
EMBEDDING_API_KEY="<key>"
```

`INDEX_DIR` is optional. The index (`vectors.npy`, `index.jsonl`, `manifest.json`) lives in `$XDG_DATA_HOME/cortex` unless it says otherwise. Point it at shared storage when another machine needs to read the index; only one machine should write it. Writes are atomic renames, so a reader sees either the old index or the new one.

## Recollection hook

Recollection is implemented as a Claude Code UserPromptSubmit hook. Put the following in a Claude Code `settings.json`.

```
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cortex",
            "args": [
              "hook",
              "recollection"
            ]
          }
        ]
      }
    ]
  }
}
```

## Cortex CLI

```
Usage: cortex [OPTIONS] COMMAND [ARGS]...

  Alpha's memory: Markdown files, with a disposable index over them.

Options:
  --help  Show this message and exit.

Commands:
  hook     Claude Code hook scripts.
  monitor  Watch for stretches of quiet, and hush the watching while it...
  reindex  Bring the index back in sync with the Markdown files on disk.
  search   Search the memories, reading the query from standard input.
  similar  Show the memories nearest to memory ID.
  store    Store a memory, reading its body from standard input.
```