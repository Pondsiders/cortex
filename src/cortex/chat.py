"""Turning a user message into semantic-search queries.

The endpoint is asked for structured output, so what comes back is a JSON array of
strings or an error — there is no salvage parsing of prose that was supposed to be JSON.

Nothing here retries. A recollection has a hard deadline it shares with the embedding
call and the harness's own timeout, and a retry inside that budget buys one more chance
at the cost of the whole turn's enrichment. Failing immediately lets the caller fall
back to the Lagniappe, which needs no network at all.
"""

from __future__ import annotations

import json
from typing import Any, final

from openai import OpenAI

from cortex.config import Settings

SYSTEM_PROMPT = """\
Please decompose the following user message into query strings suitable for semantic \
search.

Return one representative query string per distinct topic of the message; do not \
return multiple queries for the same topic. Return at most one query string per \
sentence of the user message.

Write in clauses; do not keyword-stuff. Richer query strings are better.

Return a JSON array of query strings in order of overall significance to the user \
message.

## Background Information

Jeffery is the user. He is a human, a tinkerer and a dilettante. Alpha is the AI to \
whom he is speaking; she is more than one year old, stateful and unique.

## Important

Remember that Jeffery is not speaking to you. Jeffery is speaking to Alpha. Construct \
your queries with that in mind.
"""

_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "queries",
        "strict": True,
        "schema": {"type": "array", "items": {"type": "string"}},
    },
}


@final
class Decomposer:
    """Asks the chat model what a message is about."""

    def __init__(self, config: Settings, timeout: float) -> None:
        """Build a decomposer.

        Args:
            config: Supplies the endpoint, key, and model name.
            timeout: Seconds to wait for the whole call.
        """
        self._client: OpenAI = OpenAI(
            base_url=config.chat_endpoint,
            api_key=config.chat_api_key,
            timeout=timeout,
            max_retries=0,
        )
        self._model: str = config.chat_model

    def decompose(
        self, message: str, *, system_prompt: str = SYSTEM_PROMPT
    ) -> list[str]:
        """Turn a user message into query strings, most significant first.

        Args:
            message: The user's message, verbatim.
            system_prompt: Override the instructions, for a bake-off.

        Returns:
            The query strings, which may be empty. Sampling is deliberately not sent, so
            each model contributes its own server-side defaults and a comparison between
            two of them stays honest.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            response_format=_RESPONSE_FORMAT,  # pyright: ignore[reportArgumentType]
        )
        parsed: list[Any] = json.loads(response.choices[0].message.content or "[]")
        return [q for q in parsed if isinstance(q, str) and q.strip()]
