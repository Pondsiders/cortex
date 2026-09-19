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