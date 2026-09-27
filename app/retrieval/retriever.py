from __future__ import annotations

from typing import List

from app.embedding.provider import EmbeddingProvider
from app.retrieval.models import RetrievedChunk
from app.retrieval.vector_store import VectorStore


class Retriever:
    """Embed queries and retrieve similar persisted chunks."""

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
        """Return the top_k most similar chunks for a query."""
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