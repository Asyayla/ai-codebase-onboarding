from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Set

from app.chunking.python_chunker import chunk_python_file
from app.embedding.chunk_identity import compute_chunk_id
from app.embedding.embedding_service import embed_chunks
from app.embedding.provider import EmbeddingProvider
from app.ingestion.clone import clone_repository
from app.ingestion.file_walker import walk_repository
from app.ingestion.repository_identity import compute_repository_id
from app.parsing.python_parser import parse_python_file
from app.persistence.chunk_store import ChunkStore
from app.persistence.content_hash import compute_content_hash
from app.pipeline.models import FileIndexingOutcome, RepositoryIndexingResult

_PYTHON_EXTENSIONS = {".py", ".pyi"}


class RepositoryIndexer:
    """Indexes a repository incrementally."""

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
        """Clone, process, and incrementally index a repository.

        Only new and changed chunks are embedded and persisted. Removed
        chunks are deleted only after a fully safe run.

        Args:
            source: Remote repository URL or local Git path.

        Returns:
            A summary of the indexing operation.
        """
        repository_id = compute_repository_id(source)
        stored_hashes = self._chunk_store.get_chunk_hashes(repository_id)
        clone_root = clone_repository(source)

        try:
            relative_paths = walk_repository(clone_root)
            python_paths = [
                path for path in relative_paths if path.suffix in _PYTHON_EXTENSIONS
            ]

            file_outcomes: List[FileIndexingOutcome] = []
            all_embedded = []
            all_failures = []
            all_new_ids: List[str] = []
            all_changed_ids: List[str] = []
            all_unchanged_ids: List[str] = []
            current_chunk_ids: Set[str] = set()
            run_is_safe = True

            for relative_path in python_paths:
                outcome = self._index_file(
                    clone_root, relative_path, repository_id, stored_hashes
                )
                file_outcomes.append(outcome)
                all_embedded.extend(outcome.embedded)
                all_failures.extend(outcome.embedding_failures)
                all_new_ids.extend(outcome.new_chunk_ids)
                all_changed_ids.extend(outcome.changed_chunk_ids)
                all_unchanged_ids.extend(outcome.unchanged_chunk_ids)

                if outcome.skipped_reason is not None or outcome.embedding_failures:
                    # Partial failures make deletion unsafe because missing
                    # chunks may simply be unreadable or failed to embed.
                    run_is_safe = False
                else:
                    current_chunk_ids.update(outcome.new_chunk_ids)
                    current_chunk_ids.update(outcome.changed_chunk_ids)
                    current_chunk_ids.update(outcome.unchanged_chunk_ids)

            removed_chunk_ids: List[str] = []
            if run_is_safe:
                stored_ids = set(stored_hashes.keys())
                removed_ids = stored_ids - current_chunk_ids
                if removed_ids:
                    self._chunk_store.delete_chunks(repository_id, removed_ids)
                removed_chunk_ids = sorted(removed_ids)

            return RepositoryIndexingResult(
                repository_id=repository_id,
                files_considered=len(python_paths),
                file_outcomes=file_outcomes,
                embedded=all_embedded,
                embedding_failures=all_failures,
                new_chunk_ids=all_new_ids,
                changed_chunk_ids=all_changed_ids,
                unchanged_chunk_ids=all_unchanged_ids,
                removed_chunk_ids=removed_chunk_ids,
            )
        finally:
            shutil.rmtree(clone_root, ignore_errors=True)

    def _index_file(
        self,
        clone_root: Path,
        relative_path: Path,
        repository_id: str,
        stored_hashes: Dict[str, str],
    ) -> FileIndexingOutcome:
        """Process a single Python file and return its indexing outcome."""

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
            # Empty files are valid outcomes, not errors.
            return FileIndexingOutcome(
                file_path=relative_path,
                chunks_created=0,
                skipped_reason=None,
            )

        new_chunk_ids: List[str] = []
        changed_chunk_ids: List[str] = []
        unchanged_chunk_ids: List[str] = []
        chunks_to_embed = []

        for chunk in chunks:
            # Keep classification and embedding chunk IDs consistent.
            chunk_id = compute_chunk_id(
                repository_id,
                chunk.file_path,
                chunk.symbol_type,
                chunk.symbol_name,
                chunk.start_line,
                chunk.end_line,
            )
            content_hash = compute_content_hash(chunk.content)
            stored_hash = stored_hashes.get(chunk_id)

            if stored_hash is None:
                new_chunk_ids.append(chunk_id)
                chunks_to_embed.append(chunk)
            elif stored_hash == content_hash:
                unchanged_chunk_ids.append(chunk_id)
            else:
                changed_chunk_ids.append(chunk_id)
                chunks_to_embed.append(chunk)

        embedded: List = []
        failures: List = []

        if chunks_to_embed:
            embedding_result = embed_chunks(
                chunks_to_embed,
                self._embedding_provider,
                repository_id,
                batch_size=self._batch_size,
            )
            embedded = embedding_result.embedded
            failures = embedding_result.failures

            if embedded:
                self._chunk_store.save_chunks(repository_id, embedded)

        return FileIndexingOutcome(
            file_path=relative_path,
            chunks_created=len(chunks),
            skipped_reason=None,
            embedded=embedded,
            embedding_failures=failures,
            new_chunk_ids=new_chunk_ids,
            changed_chunk_ids=changed_chunk_ids,
            unchanged_chunk_ids=unchanged_chunk_ids,
        )