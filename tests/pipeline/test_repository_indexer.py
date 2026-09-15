"""
test_repository_indexer.py

Tests for app.pipeline.repository_indexer.RepositoryIndexer.

These tests exercise the orchestrator against small, local git
repositories (created on the fly under tmp_path) rather than any real
remote - clone_repository already supports cloning from a local path
(see clone.py's --no-local design note), so no network access or mocking
of the ingestion layer is needed.

Only the two genuinely external dependencies - the embedding provider and
the chunk store - are faked. Everything else (git, the filesystem,
parsing, chunking) runs for real, exercising the real integration.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

import pytest

from app.embedding.models import EmbeddedChunk
from app.ingestion.repository_identity import compute_repository_id
from app.pipeline.repository_indexer import RepositoryIndexer
from tests.fakes.fake_chunk_store import FakeChunkStore
from tests.fakes.fake_embedding_provider import FakeEmbeddingProvider


def _make_git_repo(root: Path, files: Dict[str, bytes]) -> Path:
    """Create a local git repository under `root` containing `files`
    (relative path -> raw bytes), with a single commit. Returns the
    repository's path, suitable as a `source` for clone_repository."""
    root.mkdir(parents=True, exist_ok=True)

    for relative_path, content in files.items():
        file_path = root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True)

    return root


class _RaisingChunkStore:
    """Test-only ChunkStore double whose save_chunks always raises.
    Structurally satisfies the ChunkStore Protocol (only save_chunks is
    ever called by the orchestrator when there is something to save)."""

    def save_chunks(self, repository_id, chunks) -> None:
        raise RuntimeError("simulated persistence failure")

    def get_chunk_hashes(self, repository_id):
        return {}

    def delete_chunks(self, repository_id, chunk_ids) -> None:
        pass

    def delete_repository(self, repository_id) -> None:
        pass


class _MarkerFailingEmbeddingProvider:
    """Test-only EmbeddingProvider double that raises for any batch
    containing a chunk whose content includes _FAIL_MARKER, and otherwise
    delegates to a real FakeEmbeddingProvider. Used together with
    batch_size=1 so failure can be isolated to a single chunk."""

    FAIL_MARKER = "FAIL_MARKER"

    def __init__(self) -> None:
        self._delegate = FakeEmbeddingProvider()

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def dimensions(self) -> int:
        return self._delegate.dimensions

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if any(self.FAIL_MARKER in text for text in texts):
            raise RuntimeError("simulated embedding failure")
        return self._delegate.embed_batch(texts)


@pytest.fixture
def embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def chunk_store() -> FakeChunkStore:
    return FakeChunkStore()


def test_happy_path_single_file_is_embedded_and_persisted(tmp_path, embedding_provider, chunk_store):
    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "module_a.py": b"def foo():\n    return 1\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, chunk_store)
    result = indexer.index_repository(str(repo_dir))

    expected_repository_id = compute_repository_id(str(repo_dir))
    assert result.repository_id == expected_repository_id
    assert result.files_considered == 1
    assert len(result.file_outcomes) == 1

    outcome = result.file_outcomes[0]
    assert outcome.file_path == Path("module_a.py")
    assert outcome.skipped_reason is None
    assert outcome.chunks_created == 1
    assert len(outcome.embedded) == 1
    assert outcome.embedding_failures == []

    # The provider actually received the chunk's content.
    assert embedding_provider.received_batches == [["def foo():\n    return 1"]]

    # The chunk was actually persisted.
    embedded_chunk: EmbeddedChunk = outcome.embedded[0]
    stored_hashes = chunk_store.get_chunk_hashes(expected_repository_id)
    assert stored_hashes == {embedded_chunk.chunk_id: chunk_store.get_stored_chunk(
        expected_repository_id, embedded_chunk.chunk_id
    ).content_hash}


def test_multiple_python_files_are_all_processed(tmp_path, embedding_provider, chunk_store):
    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "module_a.py": b"def foo():\n    return 1\n",
            "module_b.py": b"def bar():\n    return 2\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, chunk_store)
    result = indexer.index_repository(str(repo_dir))

    assert result.files_considered == 2
    processed_files = {outcome.file_path for outcome in result.file_outcomes}
    assert processed_files == {Path("module_a.py"), Path("module_b.py")}
    assert all(outcome.skipped_reason is None for outcome in result.file_outcomes)
    assert len(result.embedded) == 2

    stored_hashes = chunk_store.get_chunk_hashes(result.repository_id)
    assert len(stored_hashes) == 2


def test_non_python_files_are_ignored(tmp_path, embedding_provider, chunk_store):
    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "module_a.py": b"def foo():\n    return 1\n",
            "README.md": b"# hello\n",
            "styles.js": b"console.log('hi');\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, chunk_store)
    result = indexer.index_repository(str(repo_dir))

    # Only the .py file is considered, even though walk_repository()
    # itself also returns README.md (an allow-listed doc filename).
    assert result.files_considered == 1
    assert result.file_outcomes[0].file_path == Path("module_a.py")


def test_empty_repository_with_no_python_files(tmp_path, embedding_provider, chunk_store):
    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "README.md": b"# hello\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, chunk_store)
    result = indexer.index_repository(str(repo_dir))

    assert result.files_considered == 0
    assert result.file_outcomes == []
    assert result.embedded == []
    assert result.embedding_failures == []
    assert embedding_provider.received_batches == []
    assert chunk_store.get_chunk_hashes(result.repository_id) == {}


def test_parse_failure_does_not_abort_repository(tmp_path, embedding_provider, chunk_store):
    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "broken.py": b"def broken(:\n    pass\n",
            "module_a.py": b"def foo():\n    return 1\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, chunk_store)
    result = indexer.index_repository(str(repo_dir))

    assert result.files_considered == 2
    outcomes_by_path = {outcome.file_path: outcome for outcome in result.file_outcomes}

    broken_outcome = outcomes_by_path[Path("broken.py")]
    assert broken_outcome.skipped_reason == "parse_error"
    assert broken_outcome.chunks_created == 0
    assert broken_outcome.embedded == []

    good_outcome = outcomes_by_path[Path("module_a.py")]
    assert good_outcome.skipped_reason is None
    assert good_outcome.chunks_created == 1
    assert len(good_outcome.embedded) == 1


def test_unreadable_file_does_not_abort_repository(tmp_path, embedding_provider, chunk_store):
    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            # Invalid UTF-8 byte sequence, so reading it with
            # encoding="utf-8" raises UnicodeDecodeError.
            "bad_encoding.py": b"\xff\xfe\x00bad",
            "module_a.py": b"def foo():\n    return 1\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, chunk_store)
    result = indexer.index_repository(str(repo_dir))

    assert result.files_considered == 2
    outcomes_by_path = {outcome.file_path: outcome for outcome in result.file_outcomes}

    bad_outcome = outcomes_by_path[Path("bad_encoding.py")]
    assert bad_outcome.skipped_reason == "unreadable"
    assert bad_outcome.chunks_created == 0
    assert bad_outcome.embedded == []

    good_outcome = outcomes_by_path[Path("module_a.py")]
    assert good_outcome.skipped_reason is None
    assert len(good_outcome.embedded) == 1


def test_file_with_no_chunks_skips_embedding_and_persistence(tmp_path, embedding_provider, chunk_store):
    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "empty.py": b"# just a comment\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, chunk_store)
    result = indexer.index_repository(str(repo_dir))

    assert result.files_considered == 1
    outcome = result.file_outcomes[0]
    assert outcome.skipped_reason is None
    assert outcome.chunks_created == 0
    assert outcome.embedded == []

    assert embedding_provider.received_batches == []
    assert chunk_store.get_chunk_hashes(result.repository_id) == {}


def test_embedding_failure_persists_only_successful_chunks(tmp_path, chunk_store):
    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "module_a.py": (
                b"def good_one():\n"
                b"    return 1\n"
                b"\n"
                b"\n"
                b"def bad_one():\n"
                b"    return 'FAIL_MARKER'\n"
            ),
        },
    )

    provider = _MarkerFailingEmbeddingProvider()
    # batch_size=1 isolates the failure to exactly the chunk containing
    # the marker; the other chunk is embedded in its own batch.
    indexer = RepositoryIndexer(provider, chunk_store, batch_size=1)
    result = indexer.index_repository(str(repo_dir))

    assert result.files_considered == 1
    outcome = result.file_outcomes[0]
    assert outcome.chunks_created == 2
    assert len(outcome.embedded) == 1
    assert len(outcome.embedding_failures) == 1
    assert outcome.embedded[0].symbol_name == "good_one"
    assert outcome.embedding_failures[0].symbol_name == "bad_one"

    # Only the successful chunk was persisted.
    stored_hashes = chunk_store.get_chunk_hashes(result.repository_id)
    assert len(stored_hashes) == 1
    assert outcome.embedded[0].chunk_id in stored_hashes


def test_persistence_failure_propagates(tmp_path, embedding_provider):
    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "module_a.py": b"def foo():\n    return 1\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, _RaisingChunkStore())

    with pytest.raises(RuntimeError, match="simulated persistence failure"):
        indexer.index_repository(str(repo_dir))


def test_temporary_clone_is_cleaned_up_after_success(tmp_path, monkeypatch, embedding_provider, chunk_store):
    import app.ingestion.clone as clone_module

    created_paths: List[Path] = []
    real_mkdtemp = clone_module.tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created_paths.append(Path(path))
        return path

    monkeypatch.setattr(clone_module.tempfile, "mkdtemp", _tracking_mkdtemp)

    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "module_a.py": b"def foo():\n    return 1\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, chunk_store)
    indexer.index_repository(str(repo_dir))

    assert len(created_paths) == 1
    assert not created_paths[0].exists()


def test_temporary_clone_is_cleaned_up_after_failure(tmp_path, monkeypatch, embedding_provider):
    import app.ingestion.clone as clone_module

    created_paths: List[Path] = []
    real_mkdtemp = clone_module.tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created_paths.append(Path(path))
        return path

    monkeypatch.setattr(clone_module.tempfile, "mkdtemp", _tracking_mkdtemp)

    repo_dir = _make_git_repo(
        tmp_path / "repo",
        {
            "module_a.py": b"def foo():\n    return 1\n",
        },
    )

    indexer = RepositoryIndexer(embedding_provider, _RaisingChunkStore())

    with pytest.raises(RuntimeError):
        indexer.index_repository(str(repo_dir))

    assert len(created_paths) == 1
    assert not created_paths[0].exists()