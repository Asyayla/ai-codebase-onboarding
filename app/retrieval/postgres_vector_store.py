from __future__ import annotations

import math
from typing import List

from sqlalchemy import case, func, literal, select
from sqlalchemy.exc import DBAPIError

from app.persistence.database import session_scope
from app.persistence.models import StoredChunk
from app.persistence.orm_models import ChunkRecord
from app.retrieval.models import RetrievedChunk

_DIMENSION_MISMATCH_MARKER = "different vector dimensions"


def _magnitude(vector: List[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _to_stored_chunk(record: ChunkRecord) -> StoredChunk:
    return StoredChunk(
        chunk_id=record.chunk_id,
        repository_id=record.repository_id,
        file_path=record.file_path,
        language=record.language,
        symbol_type=record.symbol_type,
        symbol_name=record.symbol_name,
        start_line=record.start_line,
        end_line=record.end_line,
        content=record.content,
        parent_symbol_name=record.parent_symbol_name,
        docstring=record.docstring,
        embedding=list(record.embedding),
        embedding_model=record.embedding_model,
        content_hash=record.content_hash,
    )


class PostgresVectorStore:
    """PostgreSQL/pgvector-backed implementation of the VectorStore protocol."""

    def search(
        self,
        repository_id: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if _magnitude(query_embedding) == 0.0:
            distance_expr = literal(1.0).label("distance")
        else:
            distance_expr = case(
                (func.vector_norm(ChunkRecord.embedding) == 0, 1.0),
                else_=ChunkRecord.embedding.cosine_distance(query_embedding),
            ).label("distance")

        stmt = (
            select(ChunkRecord, distance_expr)
            .where(ChunkRecord.repository_id == repository_id)
            .order_by(distance_expr.asc(), ChunkRecord.chunk_id.asc())
            .limit(top_k)
        )

        try:
            with session_scope() as session:
                rows = session.execute(stmt).all()
                return [
                    RetrievedChunk(
                        stored_chunk=_to_stored_chunk(record),
                        score=1.0 - distance,
                        rank=index + 1,
                    )
                    for index, (record, distance) in enumerate(rows)
                ]
        except DBAPIError as exc:
            if _DIMENSION_MISMATCH_MARKER in str(exc.orig):
                raise ValueError(
                    "query embedding dimensions do not match a stored "
                    "chunk's embedding dimensions"
                ) from exc
            raise