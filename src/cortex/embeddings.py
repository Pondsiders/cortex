"""Turning memory bodies and search queries into vectors.

Documents are embedded bare; queries are wrapped in the instruction prefix Qwen 3
Embedding's model card prescribes. Both halves live here because both are coupled to
the specific model, and if the model is ever swapped they have to be revisited
together — along with re-embedding the whole corpus.

Ember saturates at four concurrent requests — the card is compute-bound past that, and
batching inside a request only eliminates round trips, because llama-server processes a
batch serially inside one slot. So the defaults are a few lanes of modest batches;
both are options, because the right numbers depend on the endpoint of the day.

The hard limit is the endpoint's per-slot context. Ember runs the embedding model with
``--ctx-size 8192 --parallel 4``, which llama-server divides into 2,048 tokens per
request. A single document longer than that fails no matter how small the batch is.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import final

from openai import OpenAI

from cortex.config import Settings

BATCH_SIZE = 8
CONCURRENCY = 4

QUERY_TASK = (
    "Given a search query, retrieve relevant passages that are similar to the query"
)
"""The instruction Qwen 3 Embedding is given for query-side inputs."""


@dataclass(frozen=True)
class EmbeddedBatch:
    """A batch of texts and the vectors that came back, in the same order."""

    indices: Sequence[int]
    vectors: Sequence[Sequence[float]]


@final
class Embedder:
    """A thin, parallel wrapper over an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        config: Settings,
        batch_size: int = BATCH_SIZE,
        concurrency: int = CONCURRENCY,
    ) -> None:
        """Build an embedder.

        Args:
            config: Supplies the endpoint, key, and model name.
            batch_size: Texts per request.
            concurrency: Requests in flight.
        """
        self._client: OpenAI = OpenAI(
            base_url=config.embedding_endpoint,
            api_key=config.embedding_api_key,
            timeout=120.0,
            max_retries=3,
        )
        self._model: str = config.embedding_model
        self._batch_size: int = batch_size
        self._concurrency: int = concurrency

    def dimensions(self) -> int:
        """Ask the endpoint how wide its vectors are."""
        response = self._client.embeddings.create(
            model=self._model, input=["dimension probe"]
        )
        return len(response.data[0].embedding)

    def embed_query(self, query: str) -> Sequence[float]:
        """Embed one search query, with the model's instruction prefix applied.

        Args:
            query: The raw query text.

        Returns:
            The query vector.
        """
        response = self._client.embeddings.create(
            model=self._model, input=[f"Instruct: {QUERY_TASK}\nQuery:{query}"]
        )
        return response.data[0].embedding

    def _embed(self, indices: Sequence[int], texts: Sequence[str]) -> EmbeddedBatch:
        """Embed one batch, preserving order."""
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return EmbeddedBatch(
            indices=indices, vectors=[item.embedding for item in ordered]
        )

    def embed_documents(self, texts: Sequence[str]) -> Iterator[EmbeddedBatch]:
        """Embed many documents, yielding batches as they complete.

        Results arrive out of order; each batch carries the indices it came from. The
        caller is expected to commit as batches land, so that a failure partway through
        leaves the finished work durable and a re-run resumes rather than restarts.

        Args:
            texts: The document bodies, indexed by position.

        Yields:
            Completed batches, in whatever order the endpoint returns them.
        """
        batches = [
            (list(range(start, min(start + self._batch_size, len(texts)))))
            for start in range(0, len(texts), self._batch_size)
        ]
        pending: set[Future[EmbeddedBatch]] = set()
        queue = iter(batches)
        window = self._concurrency * 2

        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            while True:
                while len(pending) < window:
                    batch = next(queue, None)
                    if batch is None:
                        break
                    pending.add(
                        pool.submit(self._embed, batch, [texts[i] for i in batch])
                    )
                if not pending:
                    return
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
