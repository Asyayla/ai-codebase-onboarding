from __future__ import annotations

import os
from typing import List, Optional

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.embedding.models import EmbeddedChunk
from app.persistence import database
from app.persistence.content_hash import compute_content_hash
from app.persistence.orm_models import ChunkRecord
from app.persistence.postgres_chunk_store import PostgresChunkStore


def _make_embedded_chunk(
    *,
    chunk_id: str = "chunk-1",
    repository_id: str = "repo-1",
    file_path: str = "module.py",
    language: str = "python",
    symbol_type: Optional[str] = "function",
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
        embedding=embedding if embedding is not None else [0.1, 0.2, 0.3],
        embedding_model=embedding_model,
    )


def _fetch_records(engine, repository_id: str) -> List[ChunkRecord]:
    with Session(engine) as session:
        records = (
            session.query(ChunkRecord)
            .filter(ChunkRecord.repository_id == repository_id)
            .order_by(ChunkRecord.chunk_id)
            .all()
        )
        session.expunge_all()
        return records


def _count_all(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("AI_CODEBASE_TEST_DATABASE_URL")
    if not url:
        pytest.fail(
            "AI_CODEBASE_TEST_DATABASE_URL is not set. These tests exercise "
            "PostgresChunkStore against a real PostgreSQL database with the "
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
    """Route PostgresChunkStore's internal session_scope() at the test
    database, using database.py's own public reset_engine() hook rather
    than modifying database.py."""
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




def test_save_chunks_inserts_new_chunks(chunk_store, postgres_engine):
    chunks = [
        _make_embedded_chunk(chunk_id="chunk-1", content="def a(): pass"),
        _make_embedded_chunk(chunk_id="chunk-2", content="def b(): pass"),
    ]

    chunk_store.save_chunks("repo-1", chunks)

    records = _fetch_records(postgres_engine, "repo-1")
    assert [r.chunk_id for r in records] == ["chunk-1", "chunk-2"]
    assert records[0].content_hash == compute_content_hash("def a(): pass")
    assert records[1].content_hash == compute_content_hash("def b(): pass")


def test_save_chunks_computes_content_hash_using_shared_function(
    chunk_store, postgres_engine
):
    content = "class Foo:\n    pass\n"
    chunk_store.save_chunks("repo-1", [_make_embedded_chunk(content=content)])

    [record] = _fetch_records(postgres_engine, "repo-1")
    assert record.content_hash == compute_content_hash(content)




def test_saving_same_chunk_id_again_updates_existing_row(chunk_store, postgres_engine):
    chunk_store.save_chunks(
        "repo-1", [_make_embedded_chunk(chunk_id="chunk-1", content="old body")]
    )
    chunk_store.save_chunks(
        "repo-1", [_make_embedded_chunk(chunk_id="chunk-1", content="new body")]
    )

    records = _fetch_records(postgres_engine, "repo-1")
    assert len(records) == 1
    assert records[0].content == "new body"
    assert records[0].content_hash == compute_content_hash("new body")


def test_duplicate_chunk_id_within_single_call_uses_last_occurrence(
    chunk_store, postgres_engine
):
    chunks = [
        _make_embedded_chunk(chunk_id="chunk-1", content="first"),
        _make_embedded_chunk(chunk_id="chunk-1", content="second"),
        _make_embedded_chunk(chunk_id="chunk-1", content="last"),
    ]

    chunk_store.save_chunks("repo-1", chunks)

    records = _fetch_records(postgres_engine, "repo-1")
    assert len(records) == 1
    assert records[0].content == "last"
    assert records[0].content_hash == compute_content_hash("last")




def test_get_chunk_hashes_returns_only_requested_repository(chunk_store):
    chunk_store.save_chunks(
        "repo-a",
        [_make_embedded_chunk(chunk_id="shared", content="repo-a body")],
    )
    chunk_store.save_chunks(
        "repo-b",
        [_make_embedded_chunk(chunk_id="shared", content="repo-b body")],
    )

    hashes = chunk_store.get_chunk_hashes("repo-a")

    assert hashes == {"shared": compute_content_hash("repo-a body")}


def test_get_chunk_hashes_for_empty_repository_returns_empty_dict(chunk_store):
    assert chunk_store.get_chunk_hashes("does-not-exist") == {}




def test_delete_chunks_deletes_only_requested_repository_matching_ids(
    chunk_store, postgres_engine
):
    chunk_store.save_chunks(
        "repo-1",
        [
            _make_embedded_chunk(chunk_id="chunk-1"),
            _make_embedded_chunk(chunk_id="chunk-2"),
        ],
    )

    chunk_store.delete_chunks("repo-1", ["chunk-1"])

    records = _fetch_records(postgres_engine, "repo-1")
    assert [r.chunk_id for r in records] == ["chunk-2"]


def test_delete_chunks_does_not_affect_same_chunk_id_in_other_repository(
    chunk_store,
):
    chunk_store.save_chunks("repo-a", [_make_embedded_chunk(chunk_id="shared")])
    chunk_store.save_chunks("repo-b", [_make_embedded_chunk(chunk_id="shared")])

    chunk_store.delete_chunks("repo-a", ["shared"])

    assert chunk_store.get_chunk_hashes("repo-a") == {}
    assert "shared" in chunk_store.get_chunk_hashes("repo-b")


def test_delete_chunks_with_empty_collection_is_a_noop(chunk_store):
    chunk_store.save_chunks("repo-1", [_make_embedded_chunk(chunk_id="chunk-1")])

    chunk_store.delete_chunks("repo-1", [])

    assert "chunk-1" in chunk_store.get_chunk_hashes("repo-1")


def test_deleting_nonexistent_chunk_id_is_harmless(chunk_store):
    chunk_store.save_chunks("repo-1", [_make_embedded_chunk(chunk_id="chunk-1")])

    chunk_store.delete_chunks("repo-1", ["does-not-exist"])

    assert "chunk-1" in chunk_store.get_chunk_hashes("repo-1")




def test_delete_repository_deletes_only_that_repositorys_chunks(chunk_store):
    chunk_store.save_chunks("repo-a", [_make_embedded_chunk(chunk_id="chunk-1")])
    chunk_store.save_chunks("repo-b", [_make_embedded_chunk(chunk_id="chunk-1")])

    chunk_store.delete_repository("repo-a")

    assert chunk_store.get_chunk_hashes("repo-a") == {}
    assert "chunk-1" in chunk_store.get_chunk_hashes("repo-b")


def test_deleting_nonexistent_repository_is_harmless(chunk_store):
    chunk_store.delete_repository("does-not-exist")  # must not raise




def test_save_chunks_with_empty_list_is_a_noop(chunk_store, postgres_engine):
    chunk_store.save_chunks("repo-1", [])

    assert _count_all(postgres_engine) == 0




def test_failed_save_chunks_rolls_back_the_entire_call(chunk_store, postgres_engine):
    valid_chunk = _make_embedded_chunk(chunk_id="rollback-valid", content="fine")
    # symbol_type is NOT NULL in ChunkRecord; None forces a real database-level
    # constraint violation inside the same statement as the valid row.
    invalid_chunk = _make_embedded_chunk(
        chunk_id="rollback-invalid", symbol_type=None, content="also fine"
    )

    with pytest.raises(Exception):
        chunk_store.save_chunks("repo-1", [valid_chunk, invalid_chunk])

    # Neither row was committed: session_scope() rolled back the whole
    # transaction, including the otherwise-valid row in the same call.
    assert chunk_store.get_chunk_hashes("repo-1") == {}
    assert _count_all(postgres_engine) == 0