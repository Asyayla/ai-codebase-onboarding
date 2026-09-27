from __future__ import annotations

from typing import List, Optional

import pytest

from app.persistence.models import StoredChunk
from app.retrieval.models import RetrievedChunk
from app.retrieval.vector_store import FakeVectorStore


def _make_stored_chunk(
    chunk_id: str,
    repository_id: str = "repo-1",
    embedding: Optional[List[float]] = None,
    symbol_name: str = "foo",
    file_path: str = "module.py",
) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        repository_id=repository_id,
        file_path=file_path,
        language="python",
        symbol_type="function",
        symbol_name=symbol_name,
        start_line=1,
        end_line=2,
        content="def foo():\n    return 1",
        parent_symbol_name=None,
        docstring=None,
        embedding=embedding if embedding is not None else [1.0, 0.0, 0.0],
        embedding_model="fake-embedding-model",
        content_hash="fake-content-hash",
    )


def test_empty_store_returns_empty_list_for_valid_search():
    store = FakeVectorStore()

    results = store.search("repo-1", [1.0, 0.0, 0.0], top_k=5)

    assert results == []


def test_results_ranked_by_descending_cosine_similarity():
    store = FakeVectorStore()
    exact_match = _make_stored_chunk("exact", embedding=[1.0, 0.0, 0.0])
    partial_match = _make_stored_chunk("partial", embedding=[0.7071, 0.7071, 0.0])
    orthogonal = _make_stored_chunk("orthogonal", embedding=[0.0, 1.0, 0.0])
    store.add_chunks([orthogonal, exact_match, partial_match])

    results = store.search("repo-1", [1.0, 0.0, 0.0], top_k=3)

    assert [result.stored_chunk.chunk_id for result in results] == [
        "exact",
        "partial",
        "orthogonal",
    ]
    assert results[0].score > results[1].score > results[2].score


def test_top_k_limits_number_of_results():
    store = FakeVectorStore()
    store.add_chunks(
        [
            _make_stored_chunk("a", embedding=[1.0, 0.0, 0.0]),
            _make_stored_chunk("b", embedding=[0.9, 0.1, 0.0]),
            _make_stored_chunk("c", embedding=[0.0, 1.0, 0.0]),
        ]
    )

    results = store.search("repo-1", [1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert [result.stored_chunk.chunk_id for result in results] == ["a", "b"]


def test_repository_isolation():
    store = FakeVectorStore()
    store.add_chunk(_make_stored_chunk("repo1-chunk", repository_id="repo-1"))
    store.add_chunk(_make_stored_chunk("repo2-chunk", repository_id="repo-2"))

    results = store.search("repo-1", [1.0, 0.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0].stored_chunk.chunk_id == "repo1-chunk"


def test_invalid_top_k_raises_value_error():
    store = FakeVectorStore()
    store.add_chunk(_make_stored_chunk("a"))

    with pytest.raises(ValueError):
        store.search("repo-1", [1.0, 0.0, 0.0], top_k=0)

    with pytest.raises(ValueError):
        store.search("repo-1", [1.0, 0.0, 0.0], top_k=-1)


def test_dimension_mismatch_raises_value_error():
    store = FakeVectorStore()
    store.add_chunk(_make_stored_chunk("a", embedding=[1.0, 0.0, 0.0]))

    with pytest.raises(ValueError):
        store.search("repo-1", [1.0, 0.0, 0.0, 0.0], top_k=5)


def test_zero_magnitude_query_vector_yields_zero_similarity():
    store = FakeVectorStore()
    store.add_chunk(_make_stored_chunk("a", embedding=[1.0, 0.0, 0.0]))

    results = store.search("repo-1", [0.0, 0.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].score == 0.0


def test_zero_magnitude_stored_vector_yields_zero_similarity():
    store = FakeVectorStore()
    store.add_chunk(_make_stored_chunk("zero-vector", embedding=[0.0, 0.0, 0.0]))

    results = store.search("repo-1", [1.0, 0.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].score == 0.0


def test_returned_objects_are_retrieved_chunk_instances():
    store = FakeVectorStore()
    store.add_chunk(_make_stored_chunk("a"))

    results = store.search("repo-1", [1.0, 0.0, 0.0], top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].stored_chunk.chunk_id == "a"


def test_rank_values_are_correct_and_sequential():
    store = FakeVectorStore()
    store.add_chunks(
        [
            _make_stored_chunk("a", embedding=[1.0, 0.0, 0.0]),
            _make_stored_chunk("b", embedding=[0.9, 0.1, 0.0]),
            _make_stored_chunk("c", embedding=[0.0, 1.0, 0.0]),
        ]
    )

    results = store.search("repo-1", [1.0, 0.0, 0.0], top_k=3)

    assert [result.rank for result in results] == [1, 2, 3]


def test_tie_breaking_is_deterministic_by_insertion_order():
    store = FakeVectorStore()
    # Equal scores should preserve insertion order.
    store.add_chunk(
        _make_stored_chunk("inserted-first", embedding=[1.0, 0.0, 0.0])
    )
    store.add_chunk(
        _make_stored_chunk("inserted-second", embedding=[1.0, 0.0, 0.0])
    )

    results = store.search("repo-1", [1.0, 0.0, 0.0], top_k=5)

    assert [result.stored_chunk.chunk_id for result in results] == [
        "inserted-first",
        "inserted-second",
    ]
    assert results[0].score == results[1].score

    repeated_results = store.search("repo-1", [1.0, 0.0, 0.0], top_k=5)
    assert [result.stored_chunk.chunk_id for result in repeated_results] == [
        "inserted-first",
        "inserted-second",
    ]