from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddedChunk:
    """A successfully embedded chunk ready for persistence.

    chunk_id is deterministic and independent of chunk content.
    """

    chunk_id: str
    repository_id: str
    file_path: str
    language: str
    symbol_type: str
    symbol_name: str
    start_line: int
    end_line: int
    content: str
    parent_symbol_name: Optional[str]
    docstring: Optional[str]
    embedding: List[float]
    embedding_model: str


@dataclass
class EmbeddingFailure:
    """A chunk that could not be embedded and the reason for failure."""

    chunk_id: str
    file_path: str
    symbol_name: str
    error_message: str


@dataclass
class EmbeddingResult:
    """The successful and failed results of an embedding operation."""

    embedded: List[EmbeddedChunk] = field(default_factory=list)
    failures: List[EmbeddingFailure] = field(default_factory=list)