"""
models.py

Result structures returned by app.pipeline.repository_indexer.

These deliberately do not duplicate anything app/embedding/models.py
already provides: a successfully embedded chunk is still an
EmbeddedChunk, and a failed one is still an EmbeddingFailure. This module
only adds the two things nothing downstream already tracks - which file a
chunk came from within one repository run, and why a file was skipped
entirely (unreadable / invalid Python), if it was.

StoredChunk (app/persistence/models.py) is intentionally never referenced
here: the orchestrator's job ends at calling ChunkStore.save_chunks, not
at knowing how a store represents what it persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.embedding.models import EmbeddedChunk, EmbeddingFailure


@dataclass
class FileIndexingOutcome:
    """The outcome of processing a single discovered Python file.

    skipped_reason is None for every file that was actually parsed and
    chunked (regardless of whether it produced any chunks, and regardless
    of whether its chunks were successfully embedded) - it is only set
    when the file could not be read/decoded ("unreadable") or could not
    be parsed as valid Python ("parse_error"). A file with zero chunks,
    or with embedding failures, still has skipped_reason=None: those are
    normal outcomes, not skips.
    """

    file_path: Path
    chunks_created: int
    skipped_reason: Optional[str]
    embedded: List[EmbeddedChunk] = field(default_factory=list)
    embedding_failures: List[EmbeddingFailure] = field(default_factory=list)


@dataclass
class RepositoryIndexingResult:
    """The outcome of indexing one repository end to end.

    embedded and embedding_failures are flattened across every file, for
    callers that only care about the aggregate; file_outcomes preserves
    the per-file detail for callers (or tests) that need it.
    """

    repository_id: str
    files_considered: int
    file_outcomes: List[FileIndexingOutcome] = field(default_factory=list)
    embedded: List[EmbeddedChunk] = field(default_factory=list)
    embedding_failures: List[EmbeddingFailure] = field(default_factory=list)