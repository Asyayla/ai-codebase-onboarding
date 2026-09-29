from __future__ import annotations

from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.database import Base


class ChunkRecord(Base):
    __tablename__ = "chunks"

    __table_args__ = (
        UniqueConstraint(
            "repository_id",
            "chunk_id",
            name="uq_chunk_records_repository_chunk",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    repository_id: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_id: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    symbol_type: Mapped[str] = mapped_column(Text, nullable=False)
    symbol_name: Mapped[str] = mapped_column(Text, nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    parent_symbol_name: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    docstring: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    embedding: Mapped[List[float]] = mapped_column(
        Vector(),
        nullable=False,
    )
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)