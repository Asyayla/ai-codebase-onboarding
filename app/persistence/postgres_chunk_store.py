from __future__ import annotations

from typing import Collection, Dict, List

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from app.embedding.models import EmbeddedChunk
from app.persistence.content_hash import compute_content_hash
from app.persistence.database import session_scope
from app.persistence.orm_models import ChunkRecord

_MUTABLE_FIELDS = (
    "file_path",
    "language",
    "symbol_type",
    "symbol_name",
    "start_line",
    "end_line",
    "content",
    "parent_symbol_name",
    "docstring",
    "embedding",
    "embedding_model",
    "content_hash",
)


class PostgresChunkStore:
    """PostgreSQL-backed implementation of the ChunkStore protocol."""

    def save_chunks(self, repository_id: str, chunks: List[EmbeddedChunk]) -> None:
        if not chunks:
            return

        rows_by_chunk_id: Dict[str, dict] = {}
        for chunk in chunks:
            rows_by_chunk_id[chunk.chunk_id] = {
                "repository_id": repository_id,
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "symbol_type": chunk.symbol_type,
                "symbol_name": chunk.symbol_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
                "parent_symbol_name": chunk.parent_symbol_name,
                "docstring": chunk.docstring,
                "embedding": chunk.embedding,
                "embedding_model": chunk.embedding_model,
                "content_hash": compute_content_hash(chunk.content),
            }

        rows = list(rows_by_chunk_id.values())

        stmt = insert(ChunkRecord).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["repository_id", "chunk_id"],
            set_={field: getattr(stmt.excluded, field) for field in _MUTABLE_FIELDS},
        )

        with session_scope() as session:
            session.execute(stmt)

    def get_chunk_hashes(self, repository_id: str) -> Dict[str, str]:
        stmt = select(ChunkRecord.chunk_id, ChunkRecord.content_hash).where(
            ChunkRecord.repository_id == repository_id
        )

        with session_scope() as session:
            rows = session.execute(stmt).all()

        return {chunk_id: content_hash for chunk_id, content_hash in rows}

    def delete_chunks(
        self,
        repository_id: str,
        chunk_ids: Collection[str],
    ) -> None:
        if not chunk_ids:
            return

        stmt = delete(ChunkRecord).where(
            ChunkRecord.repository_id == repository_id,
            ChunkRecord.chunk_id.in_(chunk_ids),
        )

        with session_scope() as session:
            session.execute(stmt)

    def delete_repository(self, repository_id: str) -> None:
        stmt = delete(ChunkRecord).where(ChunkRecord.repository_id == repository_id)

        with session_scope() as session:
            session.execute(stmt)