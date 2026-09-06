"""
models.py

Data structures produced by the embedding layer (app/embedding/). Defined
separately from the code that does the actual embedding work, matching the
same pattern already used in app/parsing/models.py and
app/chunking/models.py.

EmbeddedChunk carries forward every field already present on
app/chunking's Chunk, unchanged, plus the identity and vector information
added by this layer. The embedding layer only ever adds fields - it never
drops or mutates anything chunking already produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddedChunk:
    """One chunk that has been successfully embedded, ready to be handed to
    a future persistence layer.

    chunk_id is a deterministic, content-independent identifier - see
    chunk_identity.py. It does not change when `content` changes, only
    when the chunk's identity (repository, file, symbol, location) does.
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
    """A chunk that could not be embedded, with enough information to
    identify which chunk failed and why - deliberately not a full copy of
    the chunk's content, since that isn't needed to diagnose or retry a
    failure."""

    chunk_id: str
    file_path: str
    symbol_name: str
    error_message: str


@dataclass
class EmbeddingResult:
    """The outcome of embedding a batch of chunks: whatever succeeded, and
    whatever didn't. A non-empty `failures` list is an expected, normal
    outcome - not something callers should treat as reason to discard
    `embedded`."""

    embedded: List[EmbeddedChunk] = field(default_factory=list)
    failures: List[EmbeddingFailure] = field(default_factory=list)