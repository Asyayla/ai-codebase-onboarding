from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StoredChunk:
    """A chunk persisted by a ChunkStore."""

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
    content_hash: str