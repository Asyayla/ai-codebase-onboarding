"""
Unit tests for tests/fakes/fake_embedding_provider.py.

All tests are pure/offline - no network, no disk I/O, matching the
existing test style in this project.
"""

from typing import List

import pytest

from app.embedding.provider import EmbeddingProvider
from tests.fakes.fake_embedding_provider import FakeEmbeddingProvider


def test_returns_one_vector_per_input_text():
    provider = FakeEmbeddingProvider()

    result = provider.embed_batch(["alpha", "beta", "gamma"])

    assert len(result) == 3


def test_output_order_matches_input_order():
    provider = FakeEmbeddingProvider()

    result_ab = provider.embed_batch(["alpha", "beta"])
    result_ba = provider.embed_batch(["beta", "alpha"])

    # Same two texts, opposite order -> outputs must mirror that order,
    # not just contain the same set of vectors.
    assert result_ab[0] == result_ba[1]
    assert result_ab[1] == result_ba[0]


def test_each_vector_has_exactly_the_configured_dimensions():
    provider = FakeEmbeddingProvider(dimensions=16)

    result = provider.embed_batch(["some text", "other text"])

    for vector in result:
        assert len(vector) == 16


def test_same_text_produces_the_same_vector_across_calls():
    provider = FakeEmbeddingProvider()

    first_call = provider.embed_batch(["hello world"])
    second_call = provider.embed_batch(["hello world"])

    assert first_call[0] == second_call[0]


def test_same_text_produces_the_same_vector_across_separate_instances():
    provider_one = FakeEmbeddingProvider()
    provider_two = FakeEmbeddingProvider()

    vector_one = provider_one.embed_batch(["hello world"])[0]
    vector_two = provider_two.embed_batch(["hello world"])[0]

    assert vector_one == vector_two


def test_different_texts_produce_different_vectors():
    provider = FakeEmbeddingProvider()

    result = provider.embed_batch(["def foo(): pass", "class Bar: pass"])

    assert result[0] != result[1]


def test_embed_batch_with_empty_input_returns_empty_list():
    provider = FakeEmbeddingProvider()

    result = provider.embed_batch([])

    assert result == []


def test_received_batches_starts_empty():
    provider = FakeEmbeddingProvider()

    assert provider.received_batches == []


def test_received_batches_records_multiple_calls_in_order():
    provider = FakeEmbeddingProvider()

    provider.embed_batch(["first", "call"])
    provider.embed_batch(["second", "call", "here"])

    assert provider.received_batches == [
        ["first", "call"],
        ["second", "call", "here"],
    ]


def test_received_batches_preserves_batch_boundaries():
    provider = FakeEmbeddingProvider()

    provider.embed_batch(["a", "b", "c"])
    provider.embed_batch(["d"])

    # Two separate batches of sizes 3 and 1 - NOT flattened into one list
    # of four texts.
    assert len(provider.received_batches) == 2
    assert provider.received_batches[0] == ["a", "b", "c"]
    assert provider.received_batches[1] == ["d"]


def test_received_batches_are_independent_copies_of_caller_input():
    provider = FakeEmbeddingProvider()
    original_texts: List[str] = ["one", "two"]

    provider.embed_batch(original_texts)
    original_texts.append("mutated after the call")

    assert provider.received_batches[0] == ["one", "two"]


def test_default_model_name_and_dimensions():
    provider = FakeEmbeddingProvider()

    assert provider.model_name == "fake-embedding-model"
    assert provider.dimensions == 8


def test_custom_model_name_and_dimensions():
    provider = FakeEmbeddingProvider(dimensions=32, model_name="my-test-model")

    assert provider.model_name == "my-test-model"
    assert provider.dimensions == 32


def test_zero_dimensions_raises_value_error():
    with pytest.raises(ValueError):
        FakeEmbeddingProvider(dimensions=0)


def test_negative_dimensions_raises_value_error():
    with pytest.raises(ValueError):
        FakeEmbeddingProvider(dimensions=-1)


def _embed_via_protocol_typed_parameter(provider: EmbeddingProvider, texts: List[str]) -> List[List[float]]:
    """Exercises a FakeEmbeddingProvider exactly the way real application
    code would use any EmbeddingProvider: through a parameter typed as the
    Protocol itself, calling only the methods/properties the Protocol
    defines. This is a structural-compatibility check via normal usage,
    not isinstance() - EmbeddingProvider is intentionally not
    @runtime_checkable, so isinstance() against it is not used here or
    anywhere else in this test file."""
    _ = provider.model_name
    _ = provider.dimensions
    return provider.embed_batch(texts)


def test_fake_provider_is_usable_wherever_an_embedding_provider_is_expected():
    provider = FakeEmbeddingProvider(dimensions=4, model_name="protocol-check-model")

    result = _embed_via_protocol_typed_parameter(provider, ["some text"])

    assert len(result) == 1
    assert len(result[0]) == 4