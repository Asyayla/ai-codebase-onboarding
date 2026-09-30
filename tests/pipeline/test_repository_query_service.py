from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional

import pytest
from sqlalchemy import create_engine, text

from app.ingestion.repository_identity import compute_repository_id
from app.persistence import database
from app.persistence.models import StoredChunk
from app.persistence.orm_models import ChunkRecord
from app.persistence.postgres_chunk_store import PostgresChunkStore
from app.pipeline.models import RepositoryIndexingResult
from app.pipeline.repository_query_service import RepositoryQueryService
from app.retrieval.models import RetrievedChunk
from app.retrieval.postgres_vector_store import PostgresVectorStore
from app.retrieval.vector_store import FakeVectorStore
from tests.fakes.fake_chunk_store import FakeChunkStore
from tests.fakes.fake_embedding_provider import FakeEmbeddingProvider


def _init_local_git_repo(root: Path) -> Path:
    """Create a minimal real local git repository with one Python file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "module.py").write_text("def greet():\n    return 'hello'\n")

    env_git_args = ["-c", "user.email=test@example.com", "-c", "user.name=Test"]
    subprocess.run(["git", *env_git_args, "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", *env_git_args, "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", *env_git_args, "commit", "-m", "initial commit"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return root


def _make_stored_chunk(
    *,
    chunk_id: str = "chunk-1",
    repository_id: str = "repo-1",
    embedding: Optional[List[float]] = None,
) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        repository_id=repository_id,
        file_path="module.py",
        language="python",
        symbol_type="function",
        symbol_name="greet",
        start_line=1,
        end_line=2,
        content="def greet():\n    return 'hello'\n",
        parent_symbol_name=None,
        docstring=None,
        embedding=embedding if embedding is not None else [0.1] * 8,
        embedding_model="fake-embedding-model",
        content_hash="irrelevant-for-these-tests",
    )




def test_index_repository_delegates_and_returns_indexing_result(tmp_path):
    repo_path = _init_local_git_repo(tmp_path / "repo")
    service = RepositoryQueryService(
        FakeEmbeddingProvider(), FakeChunkStore(), FakeVectorStore()
    )

    result = service.index_repository(str(repo_path))

    assert isinstance(result, RepositoryIndexingResult)
    assert result.repository_id == compute_repository_id(str(repo_path))


def test_ask_delegates_and_returns_retrieved_chunks():
    vector_store = FakeVectorStore()
    vector_store.add_chunk(_make_stored_chunk(chunk_id="chunk-1", repository_id="repo-1"))
    provider = FakeEmbeddingProvider()
    service = RepositoryQueryService(provider, FakeChunkStore(), vector_store)

    results = service.ask("repo-1", "how does greet work?", top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].stored_chunk.chunk_id == "chunk-1"
    assert results[0].rank == 1


def test_indexer_and_retriever_share_exactly_one_embedding_provider_instance(
    tmp_path,
):
    repo_path = _init_local_git_repo(tmp_path / "repo")
    provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    vector_store.add_chunk(_make_stored_chunk(chunk_id="chunk-1", repository_id="repo-1"))

    service = RepositoryQueryService(provider, FakeChunkStore(), vector_store)

    service.index_repository(str(repo_path))
    service.ask("repo-1", "how does greet work?", top_k=5)

    assert len(provider.received_batches) >= 2
    assert ["how does greet work?"] in provider.received_batches


def test_ask_forwards_opaque_repository_id_unchanged():
    vector_store = FakeVectorStore()
    opaque_id = "not-a-url-or-path-just-an-opaque-token-123"
    vector_store.add_chunk(_make_stored_chunk(chunk_id="chunk-1", repository_id=opaque_id))
    service = RepositoryQueryService(
        FakeEmbeddingProvider(), FakeChunkStore(), vector_store
    )

    results = service.ask(opaque_id, "query", top_k=5)

    assert len(results) == 1
    assert results[0].stored_chunk.repository_id == opaque_id


def test_index_then_ask_with_fakes_is_empty_because_fakes_do_not_share_storage(
    tmp_path,
):
    repo_path = _init_local_git_repo(tmp_path / "repo")
    service = RepositoryQueryService(
        FakeEmbeddingProvider(), FakeChunkStore(), FakeVectorStore()
    )

    result = service.index_repository(str(repo_path))
    results = service.ask(result.repository_id, "how does greet work?", top_k=5)

    assert results == []




@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("AI_CODEBASE_TEST_DATABASE_URL")
    if not url:
        pytest.fail(
            "AI_CODEBASE_TEST_DATABASE_URL is not set. This test exercises "
            "RepositoryQueryService against a real PostgreSQL database with "
            "the pgvector extension installed. Set "
            "AI_CODEBASE_TEST_DATABASE_URL to that database's connection URL "
            "before running this test file. DATABASE_URL (the production "
            "variable) is not used for tests."
        )
    return url


@pytest.fixture(scope="module")
def postgres_engine(test_database_url):
    engine = create_engine(test_database_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    database.Base.metadata.create_all(engine, tables=[ChunkRecord.__table__])
    yield engine
    database.Base.metadata.drop_all(engine, tables=[ChunkRecord.__table__])
    engine.dispose()


@pytest.fixture(autouse=True)
def _point_store_at_test_database(test_database_url, postgres_engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    database.reset_engine()
    yield
    database.reset_engine()


@pytest.fixture(autouse=True)
def _clean_chunks_table(postgres_engine):
    def _truncate() -> None:
        with postgres_engine.begin() as conn:
            conn.execute(text("DELETE FROM chunks"))

    _truncate()
    yield
    _truncate()


def test_index_then_ask_end_to_end_with_real_postgres_stores(tmp_path):
    repo_path = _init_local_git_repo(tmp_path / "repo")
    service = RepositoryQueryService(
        FakeEmbeddingProvider(),
        PostgresChunkStore(),
        PostgresVectorStore(),
    )

    result = service.index_repository(str(repo_path))
    results = service.ask(result.repository_id, "how does greet work?", top_k=5)

    assert len(results) >= 1
    assert all(r.stored_chunk.repository_id == result.repository_id for r in results)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))