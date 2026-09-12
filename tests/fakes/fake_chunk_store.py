"""
fake_chunk_store.py

An in-memory, offline stand-in for a real ChunkStore, for use in tests.
Structurally satisfies app.persistence.chunk_store.ChunkStore (no
inheritance needed - Protocol conformance is structural), with no
persistence beyond process lifetime, no similarity search, and no
external dependencies.
"""

from __future__ import annotations

from typing import Collection, Dict, List, Optional

from app.embedding.models import EmbeddedChunk
from app.persistence.content_hash import compute_content_hash
from app.persistence.models import StoredChunk


class FakeChunkStore:
    """In-memory ChunkStore for tests.

    Internal structure: repository_id -> chunk_id -> StoredChunk. Exists
    solely to prove the ChunkStore contract offline - not intended for
    any production use.
    """

    def __init__(self) -> None:
        self._chunks_by_repository: Dict[str, Dict[str, StoredChunk]] = {}

    def save_chunks(self, repository_id: str, chunks: List[EmbeddedChunk]) -> None:
        repository_chunks = self._chunks_by_repository.setdefault(repository_id, {})

        for chunk in chunks:
            # Duplicate chunk_ids within this call: later entries simply
            # overwrite earlier ones as this loop proceeds in order -
            # last write wins, with no special-casing required.
            repository_chunks[chunk.chunk_id] = StoredChunk(
                chunk_id=chunk.chunk_id,
                repository_id=repository_id,
                file_path=chunk.file_path,
                language=chunk.language,
                symbol_type=chunk.symbol_type,
                symbol_name=chunk.symbol_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                parent_symbol_name=chunk.parent_symbol_name,
                docstring=chunk.docstring,
                embedding=chunk.embedding,
                embedding_model=chunk.embedding_model,
                content_hash=compute_content_hash(chunk.content),
            )

    def get_chunk_hashes(self, repository_id: str) -> Dict[str, str]:
        repository_chunks = self._chunks_by_repository.get(repository_id, {})
        return {
            chunk_id: stored_chunk.content_hash
            for chunk_id, stored_chunk in repository_chunks.items()
        }

    def delete_chunks(self, repository_id: str, chunk_ids: Collection[str]) -> None:
        repository_chunks = self._chunks_by_repository.get(repository_id)
        if repository_chunks is None:
            return

        for chunk_id in chunk_ids:
            repository_chunks.pop(chunk_id, None)

    def delete_repository(self, repository_id: str) -> None:
        self._chunks_by_repository.pop(repository_id, None)

    def get_stored_chunk(self, repository_id: str, chunk_id: str) -> Optional[StoredChunk]:
        """Test-inspection helper - NOT part of the ChunkStore Protocol,
        which deliberately exposes only content hashes via
        get_chunk_hashes, not full records. Mirrors how
        FakeEmbeddingProvider.received_batches is documented as
        test-inspection state beyond the EmbeddingProvider Protocol.
        Returns the full StoredChunk for direct field verification in
        tests, or None if no such chunk is stored."""
        return self._chunks_by_repository.get(repository_id, {}).get(chunk_id)