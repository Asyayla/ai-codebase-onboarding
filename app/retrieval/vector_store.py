"""
vector_store.py

Defines the VectorStore contract that any similarity-search backend must
satisfy - an in-memory fake for tests today, a real pgvector-backed (or
other) implementation later. Nothing in this file knows about SQL,
pgvector, or any specific vector database - it exists purely so the rest
of the application can depend on a shape, not an implementation, matching
the existing pattern used by app/embedding/provider.py (EmbeddingProvider)
and app/persistence/chunk_store.py (ChunkStore).

VectorStore is a structural (Protocol) type, not an abstract base class,
matching those two existing patterns.

Scope, deliberately limited:
    This module covers similarity search only. It does NOT persist
    chunks (that is app/persistence/chunk_store.py's job) and it does NOT
    know how chunks got embedded (that is app/embedding/'s job). A
    VectorStore is handed already-embedded, already-persisted chunks by
    whatever populates it, and is asked only "which of these are most
    similar to this query embedding?".
"""

from __future__ import annotations

import math
from typing import Dict, List, Protocol

from app.persistence.models import StoredChunk
from app.retrieval.models import RetrievedChunk


class VectorStore(Protocol):
    """Contract for similarity search over already-embedded, already-
    persisted chunks, scoped by repository_id.

    Implementations are responsible for whatever storage/search mechanism
    they use (an in-memory list for tests, a real vector database for
    production) - this Protocol only constrains behavior, not how an
    implementation is constructed, configured, or populated.
    """

    def search(
        self,
        repository_id: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Return the top_k chunks most similar to query_embedding, among
        chunks belonging to repository_id only.

        Contract:
            - Only chunks stored under this exact repository_id are ever
              considered; chunks from other repositories are never
              returned, regardless of similarity.
            - Results are ranked by descending cosine similarity: the
              most similar chunk is first, with rank=1.
            - At most top_k results are returned. If fewer than top_k
              chunks exist for this repository, all of them are returned.
            - An empty repository (no chunks stored for repository_id)
              returns an empty list - this is not an error condition.
            - A query_embedding whose length does not match a candidate
              chunk's stored embedding length is a data-integrity problem,
              not a value to skip past - implementations must raise
              ValueError rather than silently ignoring the mismatched
              chunk or truncating/padding either vector.

        Args:
            repository_id: which repository to search within.
            query_embedding: the embedding vector to compare against.
            top_k: the maximum number of results to return. Must be a
                positive integer.

        Returns:
            A list of RetrievedChunk, ranked most-similar-first, of
            length at most top_k.

        Raises:
            ValueError: if top_k is not a positive integer, or if
                query_embedding's length does not match a stored chunk's
                embedding length.
        """
        ...


def _magnitude(vector: List[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Zero-magnitude vectors make cosine similarity mathematically
    undefined (division by zero) rather than merely small. Rather than
    raise for this case - which would abort an entire search over one
    degenerate stored embedding - or silently return some other value,
    this function defines similarity against a zero-magnitude vector as
    0.0: "not similar to anything, including itself." This is an explicit
    design choice, not an oversight, and is covered by a dedicated test.
    """
    magnitude_a = _magnitude(a)
    magnitude_b = _magnitude(b)

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    return dot_product / (magnitude_a * magnitude_b)


class FakeVectorStore:
    """In-memory VectorStore for tests.

    Internal structure: repository_id -> list of StoredChunk, in
    insertion order. Exists solely to prove the VectorStore contract
    offline - not intended for any production use. Deliberately exposes
    add_chunk/add_chunks for test setup; these are NOT part of the
    VectorStore Protocol (matching how FakeChunkStore's get_stored_chunk
    is documented as test-inspection state beyond the ChunkStore
    Protocol).
    """

    def __init__(self) -> None:
        self._chunks_by_repository: Dict[str, List[StoredChunk]] = {}

    def add_chunk(self, chunk: StoredChunk) -> None:
        """Test-setup helper - NOT part of the VectorStore Protocol.
        Appends `chunk` under its own repository_id, preserving insertion
        order (which matters for deterministic tie-breaking in search)."""
        self._chunks_by_repository.setdefault(chunk.repository_id, []).append(chunk)

    def add_chunks(self, chunks: List[StoredChunk]) -> None:
        """Test-setup helper - NOT part of the VectorStore Protocol.
        Equivalent to calling add_chunk for each chunk, in order."""
        for chunk in chunks:
            self.add_chunk(chunk)

    def search(
        self,
        repository_id: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        candidates = self._chunks_by_repository.get(repository_id, [])

        scored_candidates: List[tuple] = []
        for stored_chunk in candidates:
            if len(stored_chunk.embedding) != len(query_embedding):
                raise ValueError(
                    f"query embedding has {len(query_embedding)} dimensions, "
                    f"but stored chunk '{stored_chunk.chunk_id}' has an "
                    f"embedding of {len(stored_chunk.embedding)} dimensions"
                )

            score = _cosine_similarity(query_embedding, stored_chunk.embedding)
            scored_candidates.append((score, stored_chunk))

        # sorted() is guaranteed stable, and that stability holds even
        # with reverse=True - so chunks with equal scores keep their
        # original insertion order rather than an arbitrary one.
        ranked = sorted(scored_candidates, key=lambda pair: pair[0], reverse=True)

        top_results = ranked[:top_k]

        return [
            RetrievedChunk(stored_chunk=stored_chunk, score=score, rank=index + 1)
            for index, (score, stored_chunk) in enumerate(top_results)
        ]