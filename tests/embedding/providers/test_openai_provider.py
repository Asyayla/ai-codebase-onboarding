from __future__ import annotations

from typing import Any, List, Optional

import pytest

from app.embedding.providers.openai_provider import OpenAIEmbeddingProvider


class _FakeEmbeddingItem:
    def __init__(self, index: int, embedding: List[float]) -> None:
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data: List[_FakeEmbeddingItem]) -> None:
        self.data = data


class _FakeEmbeddingsResource:
    """Stands in for client.embeddings, recording calls made to .create()."""

    def __init__(
        self,
        response: Optional[_FakeEmbeddingResponse] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: List[dict] = []

    def create(self, **kwargs: Any) -> _FakeEmbeddingResponse:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _FakeOpenAIClient:
    def __init__(self, embeddings: _FakeEmbeddingsResource) -> None:
        self.embeddings = embeddings


def _response_in_order(vectors: List[List[float]]) -> _FakeEmbeddingResponse:
    return _FakeEmbeddingResponse(
        data=[_FakeEmbeddingItem(index=i, embedding=v) for i, v in enumerate(vectors)]
    )



def test_model_name_is_exactly_text_embedding_3_small():
    embeddings_resource = _FakeEmbeddingsResource(response=_response_in_order([]))
    provider = OpenAIEmbeddingProvider(client=_FakeOpenAIClient(embeddings_resource))

    assert provider.model_name == "text-embedding-3-small"


def test_dimensions_is_exactly_1536():
    embeddings_resource = _FakeEmbeddingsResource(response=_response_in_order([]))
    provider = OpenAIEmbeddingProvider(client=_FakeOpenAIClient(embeddings_resource))

    assert provider.dimensions == 1536



def test_embed_batch_sends_complete_input_list_in_one_call():
    embeddings_resource = _FakeEmbeddingsResource(
        response=_response_in_order([[0.1] * 1536, [0.2] * 1536, [0.3] * 1536])
    )
    client = _FakeOpenAIClient(embeddings_resource)
    provider = OpenAIEmbeddingProvider(client=client)

    provider.embed_batch(["alpha", "beta", "gamma"])

    assert len(embeddings_resource.calls) == 1
    assert embeddings_resource.calls[0]["input"] == ["alpha", "beta", "gamma"]


def test_embed_batch_passes_the_correct_model():
    embeddings_resource = _FakeEmbeddingsResource(response=_response_in_order([[0.1] * 1536]))
    provider = OpenAIEmbeddingProvider(client=_FakeOpenAIClient(embeddings_resource))

    provider.embed_batch(["alpha"])

    assert embeddings_resource.calls[0]["model"] == "text-embedding-3-small"


def test_embed_batch_explicitly_requests_float_encoding_format():
    embeddings_resource = _FakeEmbeddingsResource(response=_response_in_order([[0.1] * 1536]))
    provider = OpenAIEmbeddingProvider(client=_FakeOpenAIClient(embeddings_resource))

    provider.embed_batch(["alpha"])

    assert embeddings_resource.calls[0]["encoding_format"] == "float"



def test_embed_batch_returns_vectors_in_response_order_even_if_response_is_shuffled():
    # Response deliberately out of order to prove ordering is by the
    # response's own `index` field, not by list position in response.data.
    shuffled_response = _FakeEmbeddingResponse(
        data=[
            _FakeEmbeddingItem(index=2, embedding=[3.0] * 1536),
            _FakeEmbeddingItem(index=0, embedding=[1.0] * 1536),
            _FakeEmbeddingItem(index=1, embedding=[2.0] * 1536),
        ]
    )
    provider = OpenAIEmbeddingProvider(client=_FakeOpenAIClient(_FakeEmbeddingsResource(response=shuffled_response)))

    vectors = provider.embed_batch(["first", "second", "third"])

    assert vectors == [[1.0] * 1536, [2.0] * 1536, [3.0] * 1536]



def test_embed_batch_uses_the_injected_client_instance():
    embeddings_resource = _FakeEmbeddingsResource(response=_response_in_order([[0.1] * 1536]))
    injected_client = _FakeOpenAIClient(embeddings_resource)
    provider = OpenAIEmbeddingProvider(client=injected_client)

    provider.embed_batch(["alpha"])

    assert len(embeddings_resource.calls) == 1




def test_embed_batch_propagates_api_exceptions():
    class _SimulatedApiError(Exception):
        pass

    embeddings_resource = _FakeEmbeddingsResource(error=_SimulatedApiError("boom"))
    provider = OpenAIEmbeddingProvider(client=_FakeOpenAIClient(embeddings_resource))

    with pytest.raises(_SimulatedApiError):
        provider.embed_batch(["alpha"])



def test_embed_batch_with_empty_input_returns_empty_list_without_calling_api():
    embeddings_resource = _FakeEmbeddingsResource(response=_response_in_order([]))
    provider = OpenAIEmbeddingProvider(client=_FakeOpenAIClient(embeddings_resource))

    result = provider.embed_batch([])

    assert result == []
    assert embeddings_resource.calls == []
