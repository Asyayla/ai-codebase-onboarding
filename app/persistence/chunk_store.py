"""
chunk_store.py

Defines the ChunkStore contract that any persistence backend must satisfy
- an in-memory fake for tests today, a real PostgreSQL/pgvector-backed
implementation later. Nothing in this file knows about SQL, pgvector, or
any specific storage technology - it exists purely so the rest of the
application can depend on a shape, not an implementation, exactly as
app/embedding/provider.py already does for embedding backends.

ChunkStore is a structural (Protocol) type, not an abstract base class,
matching EmbeddingProvider's existing pattern in this codebase.

Scope: this Protocol covers storage and retrieval of already-embedded
chunks, scoped by repository. It does NOT include similarity/vector
search (that is a future retrieval layer's concern) and it does NOT
accept EmbeddingFailure objects (failures are not persisted - see the
Phase 3D design notes).
"""

from __future__ import annotations

from typing import Collection, Dict, List, Protocol

from app.embedding.models import EmbeddedChunk


class ChunkStore(Protocol):
    """Contract for persisting and retrieving EmbeddedChunk records,
    scoped by repository_id.

    Implementations are responsible for whatever storage mechanism they
    use (an in-memory dict for tests, a real database for production) -
    this Protocol only constrains behavior, not how an implementation is
    constructed or configured.
    """

    def save_chunks(self, repository_id: str, chunks: List[EmbeddedChunk]) -> None:
        """Insert new chunks and replace existing ones, matched by
        chunk_id.

        Contract:
            - A chunk_id not currently stored for this repository is
              inserted.
            - A chunk_id already stored for this repository has its
              stored record replaced entirely (including a freshly
              computed content_hash).
            - The caller is responsible for deciding which chunks to
              include - this method has no "skip if unchanged" logic of
              its own; every chunk passed in is upserted.
            - If `chunks` contains more than one EmbeddedChunk with the
              same chunk_id, the last one in the list wins.

        Args:
            repository_id: which repository these chunks belong to.
            chunks: the chunks to persist. May be empty.
        """
        ...

    def get_chunk_hashes(self, repository_id: str) -> Dict[str, str]:
        """Return {chunk_id: content_hash} for every chunk currently
        stored under this repository.

        Returns an empty dict if nothing has ever been stored for this
        repository_id - this is not an error condition.

        This is intentionally the only read method on ChunkStore: it is
        sufficient on its own to drive a full re-ingestion diff (which
        chunks are new, changed, unchanged, or removed) without any
        further query methods.
        """
        ...

    def delete_chunks(self, repository_id: str, chunk_ids: Collection[str]) -> None:
        """Remove the given chunk_ids from this repository, if present.

        chunk_ids that are not currently stored are silently ignored,
        not an error.
        """
        ...

    def delete_repository(self, repository_id: str) -> None:
        """Remove every chunk stored under this repository_id."""
        ...