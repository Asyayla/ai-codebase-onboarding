"""
retriever.py

Responsibility:
    Turn a user's natural-language query into ranked, retrieval-ready
    chunks, by composing two existing contracts: EmbeddingProvider (to
    embed the query) and VectorStore (to search already-persisted,
    already-embedded chunks).

    This module owns: input validation, calling embed_batch with exactly
    one text, validating the shape of the provider's response, and
    delegating the actual similarity search to VectorStore.search.

    This module does NOT know about parsing, chunking, persistence, or
    any database - those are the responsibility of earlier stages
    (app/parsing, app/chunking, app/persistence) and of whatever populates
    the VectorStore. It does NOT call an LLM and does NOT construct any
    prompt: turning retrieved chunks into an answer for a user is a
    future stage's responsibility, not this one's.
"""

from __future__ import annotations

from typing import List

from app.embedding.provider import EmbeddingProvider
from app.retrieval.models import RetrievedChunk
from app.retrieval.vector_store import VectorStore


class Retriever:
    """Embeds a query and searches for its most similar persisted chunks.

    Only the two dependencies that are genuine external side effects
    (talking to an embedding backend, talking to a vector-search backend)
    are injected. Neither is instantiated internally - the caller decides
    which concrete EmbeddingProvider and VectorStore to use, exactly as
    app.pipeline.repository_indexer.RepositoryIndexer already does for
    EmbeddingProvider and ChunkStore.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(
        self,
        repository_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Embed `query` and return the top_k most similar chunks stored
        under `repository_id`.

        Args:
            repository_id: which repository's chunks to search within.
                Passed through to VectorStore.search unchanged.
            query: the user's natural-language query. Must not be empty
                or whitespace-only.
            top_k: the maximum number of results to return. Must be a
                positive integer.

        Returns:
            Whatever VectorStore.search returns, unchanged - a list of
            RetrievedChunk ranked most-similar-first, of length at most
            top_k.

        Raises:
            ValueError: if query is empty/whitespace-only, if top_k is
                not a positive integer, if the embedding provider does
                not return exactly one vector for the single query, or if
                that vector's length does not match
                embedding_provider.dimensions.
            Exception: whatever embedding_provider.embed_batch or
                vector_store.search raises is propagated unchanged -
                neither is caught or reinterpreted here.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        vectors = self._embedding_provider.embed_batch([query])

        if len(vectors) != 1:
            raise ValueError(
                f"expected exactly one embedding vector for one query, "
                f"got {len(vectors)}"
            )

        query_embedding = vectors[0]
        expected_dimensions = self._embedding_provider.dimensions

        if len(query_embedding) != expected_dimensions:
            raise ValueError(
                f"query embedding has {len(query_embedding)} dimensions, "
                f"expected {expected_dimensions}"
            )

        return self._vector_store.search(repository_id, query_embedding, top_k)