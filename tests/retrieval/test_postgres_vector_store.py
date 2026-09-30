from __future__ import annotations

import os
from typing import List, Optional

import pytest
from sqlalchemy import create_engine, text

from app.embedding.models import EmbeddedChunk
from app.persistence import database
from app.persistence.content_hash import compute_content_hash
from app.persistence.orm_models import ChunkRecord
from app.persistence.postgres_chunk_store import PostgresChunkStore
from app.retrieval import postgres_vector_store as postgres_vector_store_module
from app.retrieval.models import RetrievedChunk
from app.retrieval.postgres_vector_store import PostgresVectorStore


def _make_embedded_chunk(
    *,
    chunk_id: str = "chunk-1",
    repository_id: str = "repo-1",
    file_path: str = "module.py",
    language: str = "python",
    symbol_type: str = "function",
    symbol_name: str = "foo",
    start_line: int = 1,
    end_line: int = 2,
    content: str = "def foo(): pass",
    parent_symbol_name: Optional[str] = None,
    docstring: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    embedding_model: str = "fake-embedding-model",
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        repository_id=repository_id,
        file_path=file_path,
        language=language,
        symbol_type=symbol_type,
        symbol_name=symbol_name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        parent_symbol_name=parent_symbol_name,
        docstring=docstring,
        embedding=embedding if embedding is not None else [1.0, 0.0, 0.0],
        embedding_model=embedding_model,
    )


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("AI_CODEBASE_TEST_DATABASE_URL")
    if not url:
        pytest.fail(
            "AI_CODEBASE_TEST_DATABASE_URL is not set. These tests exercise "
            "PostgresVectorStore against a real PostgreSQL database with the "
            "pgvector extension installed. Set AI_CODEBASE_TEST_DATABASE_URL "
            "to that database's connection URL before running this test file. "
            "DATABASE_URL (the production variable) is not used for tests."
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


@pytest.fixture
def chunk_store() -> PostgresChunkStore:
    return PostgresChunkStore()


@pytest.fixture
def vector_store() -> PostgresVectorStore:
    return PostgresVectorStore()




def test_search_on_empty_repository_returns_empty_list(vector_store):
    results = vector_store.search("does-not-exist", [1.0, 0.0, 0.0], top_k=5)

    assert results == []


def test_search_is_isolated_by_repository(chunk_store, vector_store):
    chunk_store.save_chunks(
        "repo-a", [_make_embedded_chunk(chunk_id="shared", embedding=[1.0, 0.0, 0.0])]
    )
    chunk_store.save_chunks(
        "repo-b", [_make_embedded_chunk(chunk_id="shared", embedding=[1.0, 0.0, 0.0])]
    )

    results = vector_store.search("repo-a", [1.0, 0.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].stored_chunk.repository_id == "repo-a"




def test_search_orders_by_descending_cosine_similarity(chunk_store, vector_store):
    chunk_store.save_chunks(
        "repo-1",
        [
            _make_embedded_chunk(chunk_id="parallel", embedding=[1.0, 0.0, 0.0]),
            _make_embedded_chunk(chunk_id="orthogonal", embedding=[0.0, 1.0, 0.0]),
            _make_embedded_chunk(chunk_id="opposite", embedding=[-1.0, 0.0, 0.0]),
        ],
    )

    results = vector_store.search("repo-1", [1.0, 0.0, 0.0], top_k=3)

    assert [r.stored_chunk.chunk_id for r in results] == [
        "parallel",
        "orthogonal",
        "opposite",
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)
    assert results[2].score == pytest.approx(-1.0)


def test_search_limits_results_to_top_k(chunk_store, vector_store):
    chunk_store.save_chunks(
        "repo-1",
        [
            _make_embedded_chunk(chunk_id="parallel", embedding=[1.0, 0.0, 0.0]),
            _make_embedded_chunk(chunk_id="orthogonal", embedding=[0.0, 1.0, 0.0]),
            _make_embedded_chunk(chunk_id="opposite", embedding=[-1.0, 0.0, 0.0]),
        ],
    )

    results = vector_store.search("repo-1", [1.0, 0.0, 0.0], top_k=2)

    assert [r.stored_chunk.chunk_id for r in results] == ["parallel", "orthogonal"]




def test_invalid_top_k_raises_before_any_database_access(vector_store, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("session_scope() must not be called for invalid top_k")

    monkeypatch.setattr(postgres_vector_store_module, "session_scope", _fail_if_called)

    with pytest.raises(ValueError, match="top_k must be a positive integer, got 0"):
        vector_store.search("repo-1", [1.0, 0.0, 0.0], top_k=0)




def test_zero_magnitude_query_vector_yields_zero_score_for_all(
    chunk_store, vector_store
):
    chunk_store.save_chunks(
        "repo-1",
        [_make_embedded_chunk(chunk_id="chunk-1", embedding=[1.0, 2.0, 3.0])],
    )

    results = vector_store.search("repo-1", [0.0, 0.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.0)


def test_zero_magnitude_stored_vector_yields_zero_score(chunk_store, vector_store):
    chunk_store.save_chunks(
        "repo-1",
        [_make_embedded_chunk(chunk_id="chunk-1", embedding=[0.0, 0.0, 0.0])],
    )

    results = vector_store.search("repo-1", [1.0, 0.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.0)




def test_equal_scores_break_ties_by_chunk_id_ascending(chunk_store, vector_store):
    # Saved in reverse-of-expected order to prove the tie-break is chunk_id,
    # not insertion order.
    chunk_store.save_chunks(
        "repo-1",
        [
            _make_embedded_chunk(chunk_id="b-chunk", embedding=[1.0, 0.0, 0.0]),
            _make_embedded_chunk(chunk_id="a-chunk", embedding=[1.0, 0.0, 0.0]),
        ],
    )

    results = vector_store.search("repo-1", [1.0, 0.0, 0.0], top_k=5)

    assert [r.stored_chunk.chunk_id for r in results] == ["a-chunk", "b-chunk"]




def test_returned_objects_have_expected_structure_and_ranks(chunk_store, vector_store):
    content = "def foo():\n    return 1\n"
    chunk_store.save_chunks(
        "repo-1",
        [
            _make_embedded_chunk(
                chunk_id="chunk-1", content=content, embedding=[1.0, 0.0, 0.0]
            ),
            _make_embedded_chunk(
                chunk_id="chunk-2", embedding=[0.0, 1.0, 0.0]
            ),
        ],
    )

    results = vector_store.search("repo-1", [1.0, 0.0, 0.0], top_k=5)

    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert [r.rank for r in results] == [1, 2]

    first = results[0]
    assert first.stored_chunk.chunk_id == "chunk-1"
    assert first.stored_chunk.content == content
    assert first.stored_chunk.content_hash == compute_content_hash(content)
    assert first.stored_chunk.embedding == [1.0, 0.0, 0.0]
    assert first.stored_chunk.embedding_model == "fake-embedding-model"




def test_dimension_mismatch_raises_value_error(chunk_store, vector_store):
    chunk_store.save_chunks(
        "repo-1",
        [_make_embedded_chunk(chunk_id="chunk-1", embedding=[1.0, 0.0, 0.0])],
    )

    with pytest.raises(ValueError):
        vector_store.search("repo-1", [1.0, 0.0], top_k=5)




def test_save_then_search_end_to_end(chunk_store, vector_store):
    chunks = [
        _make_embedded_chunk(
            chunk_id="alpha",
            content="def alpha(): pass",
            embedding=[1.0, 0.0, 0.0],
        ),
        _make_embedded_chunk(
            chunk_id="beta",
            content="def beta(): pass",
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    chunk_store.save_chunks("repo-1", chunks)

    results = vector_store.search("repo-1", [1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].stored_chunk.chunk_id == "alpha"
    assert results[0].stored_chunk.content_hash == compute_content_hash(
        "def alpha(): pass"
    )
    assert results[0].rank == 1