from __future__ import annotations

from typing import Collection, Dict, List, Protocol

from app.embedding.models import EmbeddedChunk


class ChunkStore(Protocol):
    """Contract for persisting and retrieving embedded chunks."""

    def save_chunks(self, repository_id: str, chunks: List[EmbeddedChunk]) -> None:
        """Insert or replace chunks by chunk_id."""
        ...

    def get_chunk_hashes(self, repository_id: str) -> Dict[str, str]:
        """Return stored content hashes keyed by chunk_id."""
        ...

    def delete_chunks(
        self,
        repository_id: str,
        chunk_ids: Collection[str],
    ) -> None:
        """Remove the given chunks from a repository."""
        ...

    def delete_repository(self, repository_id: str) -> None:
        """Remove all chunks stored for a repository."""
        ...