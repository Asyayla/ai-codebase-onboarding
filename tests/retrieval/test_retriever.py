from __future__ import annotations

from typing import List, Optional

import pytest

from app.persistence.models import StoredChunk
from app.retrieval.models import RetrievedChunk
from app.retrieval.retriever import Retriever


class _RecordingEmbeddingProvider:
    """Test-only EmbeddingProvider double."""

    def __init__(
        self,
        vectors_to_return: Optional[List[List[float]]] = None,
        dimensions: int = 3,
        model_name: str = "test-embedding-model",
        raise_error: Optional[Exception] = None,
    ) -> None:
        self._vectors_to_return = (
            vectors_to_return if vectors_to_return is not None else []
        )
        self._dimensions = dimensions
        self._model_name = model_name
        self._raise_error = raise_error
        self.received_texts: Optional[List[str]] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.received_texts = list(texts)
        if self._raise_error is not None:
            raise self._raise_error
        return self._vectors_to_return


class _RecordingVectorStore:
    """Test-only VectorStore double."""

    def __init__(
        self,
        results_to_return: Optional[List[RetrievedChunk]] = None,
        raise_error: Optional[Exception] = None,
    ) -> None:
        self._results_to_return = (
            results_to_return if results_to_return is not None else []
        )
        self._raise_error = raise_error
        self.received_call = None

    def search(
        self,
        repository_id: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[RetrievedChunk]:
        self.received_call = (repository_id, query_embedding, top_k)
        if self._raise_error is not None:
            raise self._raise_error
        return self._results_to_return


def _make_stored_chunk(chunk_id: str = "chunk-1") -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        repository_id="repo-1",
        file_path="module.py",
        language="python",
        symbol_type="function",
        symbol_name="foo",
        start_line=1,
        end_line=2,
        content="def foo():\n    return 1",
        parent_symbol_name=None,
        docstring=None,
        embedding=[1.0, 0.0, 0.0],
        embedding_model="fake-embedding-model",
        content_hash="fake-content-hash",
    )


def test_query_is_passed_to_embedding_provider_as_single_element_list():
    provider = _RecordingEmbeddingProvider(vectors_to_return=[[1.0, 0.0, 0.0]])
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    retriever.retrieve("repo-1", "find the parser")

    assert provider.received_texts == ["find the parser"]


def test_correct_query_embedding_passed_to_vector_store():
    provider = _RecordingEmbeddingProvider(vectors_to_return=[[0.1, 0.2, 0.3]])
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    retriever.retrieve("repo-1", "find the parser")

    assert vector_store.received_call[1] == [0.1, 0.2, 0.3]


def test_repository_id_passed_unchanged():
    provider = _RecordingEmbeddingProvider(vectors_to_return=[[1.0, 0.0, 0.0]])
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    retriever.retrieve("some-repo-id", "find the parser")

    assert vector_store.received_call[0] == "some-repo-id"


def test_top_k_passed_unchanged():
    provider = _RecordingEmbeddingProvider(vectors_to_return=[[1.0, 0.0, 0.0]])
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    retriever.retrieve("repo-1", "find the parser", top_k=7)

    assert vector_store.received_call[2] == 7


def test_empty_query_raises_value_error():
    provider = _RecordingEmbeddingProvider(vectors_to_return=[[1.0, 0.0, 0.0]])
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    with pytest.raises(ValueError):
        retriever.retrieve("repo-1", "")

    with pytest.raises(ValueError):
        retriever.retrieve("repo-1", "   ")


def test_invalid_top_k_raises_value_error():
    provider = _RecordingEmbeddingProvider(vectors_to_return=[[1.0, 0.0, 0.0]])
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    with pytest.raises(ValueError):
        retriever.retrieve("repo-1", "find the parser", top_k=0)

    with pytest.raises(ValueError):
        retriever.retrieve("repo-1", "find the parser", top_k=-3)


def test_provider_returning_zero_vectors_is_rejected():
    provider = _RecordingEmbeddingProvider(vectors_to_return=[])
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    with pytest.raises(ValueError):
        retriever.retrieve("repo-1", "find the parser")

    assert vector_store.received_call is None


def test_provider_returning_multiple_vectors_is_rejected():
    provider = _RecordingEmbeddingProvider(
        vectors_to_return=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    with pytest.raises(ValueError):
        retriever.retrieve("repo-1", "find the parser")

    assert vector_store.received_call is None


def test_query_embedding_dimension_mismatch_is_rejected():
    provider = _RecordingEmbeddingProvider(
        vectors_to_return=[[1.0, 0.0]], dimensions=3
    )
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    with pytest.raises(ValueError):
        retriever.retrieve("repo-1", "find the parser")

    assert vector_store.received_call is None


def test_vector_store_results_are_returned_unchanged():
    expected_results = [
        RetrievedChunk(stored_chunk=_make_stored_chunk("chunk-1"), score=0.9, rank=1),
        RetrievedChunk(stored_chunk=_make_stored_chunk("chunk-2"), score=0.5, rank=2),
    ]
    provider = _RecordingEmbeddingProvider(vectors_to_return=[[1.0, 0.0, 0.0]])
    vector_store = _RecordingVectorStore(results_to_return=expected_results)
    retriever = Retriever(provider, vector_store)

    results = retriever.retrieve("repo-1", "find the parser")

    assert results is expected_results


def test_provider_failure_propagates():
    provider = _RecordingEmbeddingProvider(
        raise_error=RuntimeError("provider exploded")
    )
    vector_store = _RecordingVectorStore()
    retriever = Retriever(provider, vector_store)

    with pytest.raises(RuntimeError, match="provider exploded"):
        retriever.retrieve("repo-1", "find the parser")


def test_vector_store_failure_propagates():
    provider = _RecordingEmbeddingProvider(vectors_to_return=[[1.0, 0.0, 0.0]])
    vector_store = _RecordingVectorStore(
        raise_error=RuntimeError("store exploded")
    )
    retriever = Retriever(provider, vector_store)

    with pytest.raises(RuntimeError, match="store exploded"):
        retriever.retrieve("repo-1", "find the parser")