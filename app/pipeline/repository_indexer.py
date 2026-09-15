"""
repository_indexer.py

Responsibility:
    Orchestrate the existing, independently-tested stages (ingestion,
    parsing, chunking, embedding, persistence) into the single end-to-end
    operation this project needs: "given a repository source, index every
    Python file in it."

    This module contains no parsing logic, no chunking logic, no embedding
    logic, and no persistence logic of its own - all of that already
    exists and is reused here unchanged. This module is sequencing and
    per-file failure isolation, nothing else.

    Deliberately NOT used here: app.ingestion.repository_ingestion's
    ingest_repository(). That function discards the temporary clone root
    it creates, and this orchestrator needs that root to read file
    contents. Rather than change ingest_repository's existing, tested
    contract, this module calls clone_repository() and walk_repository()
    directly - both already public and already independently tested.

Contract:
    RepositoryIndexer(embedding_provider, chunk_store, batch_size=16)
    index_repository(source: str) -> RepositoryIndexingResult

    - Only .py and .pyi files (from whatever walk_repository returns) are
      processed; walk_repository's broader allow-list exists for future
      languages, not this pipeline.
    - A single bad file (unreadable, or invalid Python) never aborts the
      repository run - it is recorded on that file's FileIndexingOutcome
      and processing continues.
    - Embedding failures are handled by the existing embed_chunks/
      EmbeddingResult contract: successful chunks are still persisted,
      failed ones are recorded and never passed to chunk_store.save_chunks.
    - Cloning failure (RepositoryCloneError) and persistence failure
      (whatever chunk_store.save_chunks raises) both propagate unchanged -
      neither is caught or reinterpreted here.
    - The temporary directory created by clone_repository() is always
      removed when index_repository() returns, whether it returns
      normally or via an exception. clone_repository() always clones into
      a brand-new temporary directory (see clone.py), even when `source`
      is itself a local path - so this cleanup never touches anything the
      caller passed in, only the copy this call created.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from app.chunking.python_chunker import chunk_python_file
from app.embedding.embedding_service import embed_chunks
from app.embedding.provider import EmbeddingProvider
from app.ingestion.clone import clone_repository
from app.ingestion.file_walker import walk_repository
from app.ingestion.repository_identity import compute_repository_id
from app.parsing.python_parser import parse_python_file
from app.persistence.chunk_store import ChunkStore
from app.pipeline.models import FileIndexingOutcome, RepositoryIndexingResult

_PYTHON_EXTENSIONS = {".py", ".pyi"}


class RepositoryIndexer:
    """Orchestrates ingestion -> parsing -> chunking -> embedding ->
    persistence for one repository source.

    Only the two dependencies that are genuine external side effects
    (talking to an embedding backend, talking to a persistence backend)
    are injected. The parser, chunker, and filesystem/ingestion functions
    are pure or dependency-free, and are used directly - injecting them
    would not improve testability, only add indirection.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunk_store: ChunkStore,
        batch_size: int = 16,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._chunk_store = chunk_store
        self._batch_size = batch_size

    def index_repository(self, source: str) -> RepositoryIndexingResult:
        """
        Clone `source`, discover its Python files, and parse/chunk/embed/
        persist each one.

        Args:
            source: anything clone_repository accepts - a remote URL or a
                local path that git can clone from.

        Returns:
            A RepositoryIndexingResult describing what was found,
            embedded, and persisted - including any per-file failures
            encountered along the way.

        Raises:
            RepositoryCloneError: propagated unchanged if cloning `source`
                fails; no partial result is meaningful at that point.
            Exception: whatever chunk_store.save_chunks raises on a
                persistence failure is propagated unchanged - ChunkStore
                currently defines no failure contract of its own to
                translate this into.
        """
        repository_id = compute_repository_id(source)
        clone_root = clone_repository(source)

        try:
            relative_paths = walk_repository(clone_root)
            python_paths = [
                path for path in relative_paths if path.suffix in _PYTHON_EXTENSIONS
            ]

            file_outcomes: List[FileIndexingOutcome] = []
            all_embedded = []
            all_failures = []

            for relative_path in python_paths:
                outcome = self._index_file(clone_root, relative_path, repository_id)
                file_outcomes.append(outcome)
                all_embedded.extend(outcome.embedded)
                all_failures.extend(outcome.embedding_failures)

            return RepositoryIndexingResult(
                repository_id=repository_id,
                files_considered=len(python_paths),
                file_outcomes=file_outcomes,
                embedded=all_embedded,
                embedding_failures=all_failures,
            )
        finally:
            shutil.rmtree(clone_root, ignore_errors=True)

    def _index_file(
        self,
        clone_root: Path,
        relative_path: Path,
        repository_id: str,
    ) -> FileIndexingOutcome:
        """Process a single Python file already known to exist under
        clone_root. Never raises for a bad file - read failures and parse
        failures are both represented as a skipped outcome instead."""
        absolute_path = clone_root / relative_path

        try:
            source_code = absolute_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return FileIndexingOutcome(
                file_path=relative_path,
                chunks_created=0,
                skipped_reason="unreadable",
            )

        parsed_file = parse_python_file(source_code, str(relative_path))
        if parsed_file.parse_errors:
            return FileIndexingOutcome(
                file_path=relative_path,
                chunks_created=0,
                skipped_reason="parse_error",
            )

        chunks = chunk_python_file(parsed_file, source_code, language="python")
        if not chunks:
            # A normal outcome (e.g. an empty file, or one with only
            # comments/blank lines) - not a skip, not an error.
            return FileIndexingOutcome(
                file_path=relative_path,
                chunks_created=0,
                skipped_reason=None,
            )

        embedding_result = embed_chunks(
            chunks,
            self._embedding_provider,
            repository_id,
            batch_size=self._batch_size,
        )

        if embedding_result.embedded:
            self._chunk_store.save_chunks(repository_id, embedding_result.embedded)

        return FileIndexingOutcome(
            file_path=relative_path,
            chunks_created=len(chunks),
            skipped_reason=None,
            embedded=embedding_result.embedded,
            embedding_failures=embedding_result.failures,
        )