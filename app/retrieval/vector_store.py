from __future__ import annotations

import math
from typing import Dict, List, Protocol

from app.persistence.models import StoredChunk
from app.retrieval.models import RetrievedChunk


class VectorStore(Protocol):
    """Contract for similarity search over stored embeddings."""

    def search(
        self,
        repository_id: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Return the most similar chunks for a repository."""
        ...


def _magnitude(vector: List[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return cosine similarity between two equal-length vectors."""
    magnitude_a = _magnitude(a)
    magnitude_b = _magnitude(b)

    # Cosine similarity is undefined for zero-magnitude vectors.
    return 0.0 if magnitude_a == 0.0 or magnitude_b == 0.0 else (
        sum(x * y for x, y in zip(a, b)) / (magnitude_a * magnitude_b)
    )


class FakeVectorStore:
    """In-memory VectorStore implementation for tests."""

    def __init__(self) -> None:
        self._chunks_by_repository: Dict[str, List[StoredChunk]] = {}

    def add_chunk(self, chunk: StoredChunk) -> None:
        """Add a chunk for test setup."""
        self._chunks_by_repository.setdefault(chunk.repository_id, []).append(chunk)

    def add_chunks(self, chunks: List[StoredChunk]) -> None:
        """Add multiple chunks for test setup."""
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

        # sorted() is stable, so equal scores preserve insertion order.
        ranked = sorted(
            scored_candidates,
            key=lambda pair: pair[0],
            reverse=True,
        )

        top_results = ranked[:top_k]

        return [
            RetrievedChunk(
                stored_chunk=stored_chunk,
                score=score,
                rank=index + 1,
            )
            for index, (score, stored_chunk) in enumerate(top_results)
        ]