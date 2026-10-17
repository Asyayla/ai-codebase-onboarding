from types import SimpleNamespace

import pytest

from app.embedding.providers.gemini_provider import GeminiEmbeddingProvider


class FakeModels:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected embedding API call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses=None):
        self.models = FakeModels(responses)


def embedding_response(*vectors):
    return SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=vector)
            for vector in vectors
        ]
    )


def test_provider_exposes_model_and_dimensions():
    provider = GeminiEmbeddingProvider(client=FakeClient())

    assert provider.model_name == "gemini-embedding-001"
    assert provider.dimensions == 1536


def test_embed_batch_uses_document_task_and_preserves_order():
    first = [0.1] * 1536
    second = [0.2] * 1536
    client = FakeClient([embedding_response(first, second)])
    provider = GeminiEmbeddingProvider(client=client)

    result = provider.embed_batch(["def first(): pass", "def second(): pass"])

    assert result == [first, second]
    call = client.models.calls[0]
    assert call["model"] == "gemini-embedding-001"
    assert call["contents"] == ["def first(): pass", "def second(): pass"]
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call["config"].output_dimensionality == 1536


def test_embed_batch_empty_input_does_not_call_api():
    client = FakeClient()
    provider = GeminiEmbeddingProvider(client=client)

    assert provider.embed_batch([]) == []
    assert client.models.calls == []


def test_embed_batch_rejects_wrong_number_of_vectors():
    client = FakeClient([embedding_response([0.1] * 1536)])
    provider = GeminiEmbeddingProvider(client=client)

    with pytest.raises(ValueError, match="expected 2 embedding vectors, got 1"):
        provider.embed_batch(["first", "second"])


def test_embed_query_uses_code_retrieval_query_task():
    vector = [0.3] * 1536
    client = FakeClient([embedding_response(vector)])
    provider = GeminiEmbeddingProvider(client=client)

    result = provider.embed_query("Where is the greeting function defined?")

    assert result == vector
    call = client.models.calls[0]
    assert call["contents"] == "Where is the greeting function defined?"
    assert call["config"].task_type == "CODE_RETRIEVAL_QUERY"
    assert call["config"].output_dimensionality == 1536


def test_embed_query_rejects_missing_vector():
    client = FakeClient([SimpleNamespace(embeddings=[])])
    provider = GeminiEmbeddingProvider(client=client)

    with pytest.raises(ValueError, match="got 0"):
        provider.embed_query("question")


def test_embed_query_rejects_wrong_vector_dimensions():
    client = FakeClient([embedding_response([0.1, 0.2])])
    provider = GeminiEmbeddingProvider(client=client)

    with pytest.raises(ValueError, match="has 2 dimensions, expected 1536"):
        provider.embed_query("question")


def test_embedding_api_errors_propagate():
    client = FakeClient([RuntimeError("API unavailable")])
    provider = GeminiEmbeddingProvider(client=client)

    with pytest.raises(RuntimeError, match="API unavailable"):
        provider.embed_query("question")
