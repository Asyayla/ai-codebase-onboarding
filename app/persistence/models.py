"""
models.py

The data structure a ChunkStore persists. Defined separately from
app/embedding/models.py because StoredChunk is a persistence-specific
shape: it carries every field already present on EmbeddedChunk, unchanged,
plus content_hash - a field the embedding layer deliberately does not know
about (see content_hash.py). This mirrors how app/parsing/models.py and
app/chunking/models.py stay separate despite one flowing into the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StoredChunk:
    """One chunk as persisted by a ChunkStore.

    Every field except content_hash is copied unchanged from the
    EmbeddedChunk that was saved. content_hash is computed by the store
    itself at save time (see content_hash.py) and is used only for
    re-ingestion change detection - it is content-DEPENDENT by design,
    unlike chunk_id.
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
    content_hash: str