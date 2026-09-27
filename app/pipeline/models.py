from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.embedding.models import EmbeddedChunk, EmbeddingFailure


@dataclass
class FileIndexingOutcome:
    """Result of processing a single Python file."""

    file_path: Path
    chunks_created: int
    skipped_reason: Optional[str]
    embedded: List[EmbeddedChunk] = field(default_factory=list)
    embedding_failures: List[EmbeddingFailure] = field(default_factory=list)
    new_chunk_ids: List[str] = field(default_factory=list)
    changed_chunk_ids: List[str] = field(default_factory=list)
    unchanged_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class RepositoryIndexingResult:
    """The outcome of indexing one repository end to end."""

    repository_id: str
    files_considered: int
    file_outcomes: List[FileIndexingOutcome] = field(default_factory=list)
    embedded: List[EmbeddedChunk] = field(default_factory=list)
    embedding_failures: List[EmbeddingFailure] = field(default_factory=list)
    new_chunk_ids: List[str] = field(default_factory=list)
    changed_chunk_ids: List[str] = field(default_factory=list)
    unchanged_chunk_ids: List[str] = field(default_factory=list)
    removed_chunk_ids: List[str] = field(default_factory=list)