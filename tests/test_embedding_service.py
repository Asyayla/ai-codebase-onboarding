from typing import List

import pytest

from app.chunking.models import Chunk
from app.embedding.chunk_identity import compute_chunk_id
from app.embedding.embedding_service import embed_chunks
from tests.fakes.fake_embedding_provider import FakeEmbeddingProvider


def _make_chunk(
    file_path: str = "app/foo.py",
    language: str = "python",
    symbol_type: str = "function",
    symbol_name: str = "foo",
    start_line: int = 1,
    end_line: int = 5,
    content: str = "def foo(): pass",
    parent_symbol_name=None,
    docstring=None,
) -> Chunk:
    return Chunk(
        file_path=file_path,
        language=language,
        symbol_type=symbol_type,
        symbol_name=symbol_name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        parent_symbol_name=parent_symbol_name,
        docstring=docstring,
    )



# Local, minimal, single-purpose broken provider doubles.
# Each exists to exercise exactly one failure mode of embed_chunks - kept
# local to this test file, per the approved design, rather than extending
# the shared FakeEmbeddingProvider.



class _WrongCountProvider:
    """Always returns one fewer vector than the number of input texts
    (unless given zero texts, in which case it returns [])."""

    def __init__(self, dimensions: int = 8, model_name: str = "wrong-count-model"):
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [[0.0] * self._dimensions for _ in texts[:-1]]


class _WrongDimensionProvider:
    """Returns the correct number of vectors, but the vector at
    `bad_index` has the wrong length. All other vectors are well-formed
    (via an internal FakeEmbeddingProvider, unmodified)."""

    def __init__(self, bad_index: int = 0, dimensions: int = 8, model_name: str = "wrong-dimension-model"):
        self._delegate = FakeEmbeddingProvider(dimensions=dimensions, model_name=model_name)
        self._bad_index = bad_index

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def dimensions(self) -> int:
        return self._delegate.dimensions

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self._delegate.embed_batch(texts)
        if 0 <= self._bad_index < len(vectors):
            vectors[self._bad_index] = vectors[self._bad_index] + [0.0]  # one extra value
        return vectors


class _RaisingProvider:
    """Always raises on embed_batch."""

    def __init__(self, dimensions: int = 8, model_name: str = "raising-model", error_message: str = "simulated provider failure"):
        self._dimensions = dimensions
        self._model_name = model_name
        self._error_message = error_message

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise RuntimeError(self._error_message)


class _FailsOnCallNumberProvider:
    """Delegates to a real (unmodified) FakeEmbeddingProvider for every
    call except the given 1-indexed call number, on which it raises.
    Used to simulate one batch failing while others succeed."""

    def __init__(self, fail_on_call_number: int, dimensions: int = 8, model_name: str = "flaky-model"):
        self._delegate = FakeEmbeddingProvider(dimensions=dimensions, model_name=model_name)
        self._fail_on_call_number = fail_on_call_number
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def dimensions(self) -> int:
        return self._delegate.dimensions

    @property
    def received_batches(self) -> List[List[str]]:
        return self._delegate.received_batches

    @property
    def call_count(self) -> int:
        """Total number of times embed_batch was invoked on this
        provider, including the call(s) that raised. Unlike
        received_batches (which only reflects calls that reached the
        underlying delegate successfully), this counts every attempt."""
        return self._call_count

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._call_count += 1
        if self._call_count == self._fail_on_call_number:
            raise RuntimeError(f"simulated failure on call {self._call_count}")
        return self._delegate.embed_batch(texts)



# 1. Empty input


def test_empty_input_returns_empty_result_and_makes_no_provider_call():
    provider = FakeEmbeddingProvider()

    result = embed_chunks([], provider, repository_id="repo-1")

    assert result.embedded == []
    assert result.failures == []
    assert provider.received_batches == []



# 2. One chunk


def test_one_chunk_produces_one_embedded_chunk_with_correct_identity():
    provider = FakeEmbeddingProvider(model_name="test-model")
    chunk = _make_chunk()

    result = embed_chunks([chunk], provider, repository_id="repo-1")

    assert len(result.embedded) == 1
    assert result.failures == []

    embedded_chunk = result.embedded[0]
    expected_chunk_id = compute_chunk_id(
        "repo-1", chunk.file_path, chunk.symbol_type, chunk.symbol_name,
        chunk.start_line, chunk.end_line,
    )
    assert embedded_chunk.chunk_id == expected_chunk_id
    assert embedded_chunk.repository_id == "repo-1"
    assert embedded_chunk.embedding_model == "test-model"



# 3. Metadata preservation


def test_every_chunk_field_is_copied_exactly():
    provider = FakeEmbeddingProvider()
    chunk = _make_chunk(
        file_path="app/services/order_service.py",
        language="python",
        symbol_type="method",
        symbol_name="OrderService.calculate_total",
        start_line=42,
        end_line=57,
        content="def calculate_total(self): ...",
        parent_symbol_name="OrderService",
        docstring="Calculates the order total.",
    )

    result = embed_chunks([chunk], provider, repository_id="repo-1")
    embedded_chunk = result.embedded[0]

    assert embedded_chunk.file_path == chunk.file_path
    assert embedded_chunk.language == chunk.language
    assert embedded_chunk.symbol_type == chunk.symbol_type
    assert embedded_chunk.symbol_name == chunk.symbol_name
    assert embedded_chunk.start_line == chunk.start_line
    assert embedded_chunk.end_line == chunk.end_line
    assert embedded_chunk.content == chunk.content
    assert embedded_chunk.parent_symbol_name == chunk.parent_symbol_name
    assert embedded_chunk.docstring == chunk.docstring



# 4. Only content is sent


def test_only_chunk_content_is_sent_to_the_provider():
    provider = FakeEmbeddingProvider()
    chunk = _make_chunk(
        file_path="secret_file_path_marker.py",
        symbol_name="secret_symbol_name_marker",
        docstring="secret_docstring_marker",
        content="def foo(): pass",
    )

    embed_chunks([chunk], provider, repository_id="repo-1")

    assert provider.received_batches == [["def foo(): pass"]]
    sent_text = provider.received_batches[0][0]
    assert "secret_file_path_marker" not in sent_text
    assert "secret_symbol_name_marker" not in sent_text
    assert "secret_docstring_marker" not in sent_text



# 5. Multiple chunks in one batch


def test_multiple_chunks_fitting_in_one_batch_makes_one_provider_call_in_order():
    provider = FakeEmbeddingProvider()
    chunks = [
        _make_chunk(symbol_name="a", content="content-a"),
        _make_chunk(symbol_name="b", content="content-b"),
        _make_chunk(symbol_name="c", content="content-c"),
    ]

    result = embed_chunks(chunks, provider, repository_id="repo-1", batch_size=16)

    assert len(provider.received_batches) == 1
    assert provider.received_batches[0] == ["content-a", "content-b", "content-c"]
    assert [c.symbol_name for c in result.embedded] == ["a", "b", "c"]



# 6. Multiple batches


def test_multiple_batches_have_correct_boundaries_and_call_count():
    provider = FakeEmbeddingProvider()
    chunks = [_make_chunk(symbol_name=f"chunk_{i}", content=f"content-{i}") for i in range(5)]

    embed_chunks(chunks, provider, repository_id="repo-1", batch_size=2)

    assert len(provider.received_batches) == 3
    assert provider.received_batches[0] == ["content-0", "content-1"]
    assert provider.received_batches[1] == ["content-2", "content-3"]
    assert provider.received_batches[2] == ["content-4"]



# 7. Exact batch boundary


def test_exact_batch_boundary_produces_exactly_the_expected_call_count():
    provider = FakeEmbeddingProvider()
    chunks = [_make_chunk(symbol_name=f"chunk_{i}") for i in range(4)]

    embed_chunks(chunks, provider, repository_id="repo-1", batch_size=2)

    assert len(provider.received_batches) == 2
    assert len(provider.received_batches[0]) == 2
    assert len(provider.received_batches[1]) == 2



# 8. batch_size > len(chunks)


def test_batch_size_larger_than_chunk_count_makes_exactly_one_call():
    provider = FakeEmbeddingProvider()
    chunks = [_make_chunk(symbol_name=f"chunk_{i}") for i in range(3)]

    embed_chunks(chunks, provider, repository_id="repo-1", batch_size=100)

    assert len(provider.received_batches) == 1
    assert len(provider.received_batches[0]) == 3



# 9. batch_size == 1


def test_batch_size_one_makes_one_call_per_chunk():
    provider = FakeEmbeddingProvider()
    chunks = [_make_chunk(symbol_name=f"chunk_{i}") for i in range(3)]

    embed_chunks(chunks, provider, repository_id="repo-1", batch_size=1)

    assert len(provider.received_batches) == 3
    for batch in provider.received_batches:
        assert len(batch) == 1



# 10. Invalid batch_size


def test_batch_size_zero_raises_value_error_with_no_provider_call():
    provider = FakeEmbeddingProvider()
    chunk = _make_chunk()

    with pytest.raises(ValueError):
        embed_chunks([chunk], provider, repository_id="repo-1", batch_size=0)

    assert provider.received_batches == []


def test_negative_batch_size_raises_value_error_with_no_provider_call():
    provider = FakeEmbeddingProvider()
    chunk = _make_chunk()

    with pytest.raises(ValueError):
        embed_chunks([chunk], provider, repository_id="repo-1", batch_size=-1)

    assert provider.received_batches == []



# 11. Deterministic chunk IDs


def test_identical_input_produces_identical_chunk_ids_across_calls():
    provider_one = FakeEmbeddingProvider()
    provider_two = FakeEmbeddingProvider()
    chunk = _make_chunk()

    result_one = embed_chunks([chunk], provider_one, repository_id="repo-1")
    result_two = embed_chunks([chunk], provider_two, repository_id="repo-1")

    assert result_one.embedded[0].chunk_id == result_two.embedded[0].chunk_id



# 12. Provider model name


def test_provider_model_name_is_copied_to_embedding_model():
    provider = FakeEmbeddingProvider(model_name="a-very-specific-model-name")
    chunk = _make_chunk()

    result = embed_chunks([chunk], provider, repository_id="repo-1")

    assert result.embedded[0].embedding_model == "a-very-specific-model-name"



# 13. Global output ordering


def test_global_output_order_is_preserved_across_multiple_batches():
    provider = FakeEmbeddingProvider()
    chunks = [_make_chunk(symbol_name=name) for name in ["e", "a", "c", "b", "d"]]

    result = embed_chunks(chunks, provider, repository_id="repo-1", batch_size=2)

    assert [c.symbol_name for c in result.embedded] == ["e", "a", "c", "b", "d"]



# 14. Provider call ordering


def test_provider_call_ordering_matches_original_chunk_order():
    provider = FakeEmbeddingProvider()
    chunks = [_make_chunk(symbol_name=f"chunk_{i}", content=f"content-{i}") for i in range(4)]

    embed_chunks(chunks, provider, repository_id="repo-1", batch_size=2)

    assert provider.received_batches == [
        ["content-0", "content-1"],
        ["content-2", "content-3"],
    ]



# 15. Wrong vector count


def test_wrong_vector_count_fails_the_entire_batch_with_correct_metadata():
    good_provider = FakeEmbeddingProvider()
    bad_provider = _WrongCountProvider()
    good_chunks = [_make_chunk(symbol_name=f"good_{i}", file_path="good.py") for i in range(2)]
    bad_chunks = [_make_chunk(symbol_name=f"bad_{i}", file_path="bad.py") for i in range(2)]

    # Process the "bad" batch through the wrong-count provider directly to
    # isolate and verify its failure behavior.
    result = embed_chunks(bad_chunks, bad_provider, repository_id="repo-1", batch_size=2)

    assert result.embedded == []
    assert len(result.failures) == 2
    for failure, chunk in zip(result.failures, bad_chunks):
        assert failure.file_path == chunk.file_path
        assert failure.symbol_name == chunk.symbol_name
        assert "2" in failure.error_message  # expected count
        assert "1" in failure.error_message  # actual count

    # Confirm a well-formed provider is unaffected by the above - i.e. the
    # failure is specific to the bad provider, not a global effect.
    good_result = embed_chunks(good_chunks, good_provider, repository_id="repo-1", batch_size=2)
    assert len(good_result.embedded) == 2
    assert good_result.failures == []


def test_wrong_vector_count_in_one_batch_does_not_prevent_other_batches():
    """A provider that fails the count check on its FIRST call, but would
    behave correctly afterward, still needs later batches processed."""

    class _WrongCountThenFineProvider:
        def __init__(self):
            self._delegate = FakeEmbeddingProvider()
            self._call_count = 0

        @property
        def model_name(self):
            return self._delegate.model_name

        @property
        def dimensions(self):
            return self._delegate.dimensions

        def embed_batch(self, texts):
            self._call_count += 1
            if self._call_count == 1:
                return [[0.0] * self.dimensions for _ in texts[:-1]]  # one short
            return self._delegate.embed_batch(texts)

    provider = _WrongCountThenFineProvider()
    chunks = [_make_chunk(symbol_name=f"chunk_{i}") for i in range(4)]

    result = embed_chunks(chunks, provider, repository_id="repo-1", batch_size=2)

    assert len(result.failures) == 2  # first batch: both chunks failed
    assert len(result.embedded) == 2  # second batch: both chunks succeeded



# 16. Wrong vector dimension


def test_wrong_vector_dimension_fails_only_that_chunk_not_its_siblings():
    provider = _WrongDimensionProvider(bad_index=1, dimensions=8)
    chunks = [_make_chunk(symbol_name=f"chunk_{i}") for i in range(3)]

    result = embed_chunks(chunks, provider, repository_id="repo-1", batch_size=3)

    assert len(result.embedded) == 2
    assert len(result.failures) == 1
    assert result.failures[0].symbol_name == "chunk_1"
    assert "8" in result.failures[0].error_message  # expected dimension
    assert "9" in result.failures[0].error_message  # actual dimension (8 + 1)
    embedded_names = {c.symbol_name for c in result.embedded}
    assert embedded_names == {"chunk_0", "chunk_2"}



# 17. Provider exception


def test_provider_exception_converts_whole_batch_to_failures_without_propagating():
    provider = _RaisingProvider(error_message="rate limit exceeded")
    chunks = [_make_chunk(symbol_name=f"chunk_{i}") for i in range(3)]

    result = embed_chunks(chunks, provider, repository_id="repo-1", batch_size=3)

    assert result.embedded == []
    assert len(result.failures) == 3
    for failure in result.failures:
        assert failure.error_message == "rate limit exceeded"


def test_provider_exception_in_one_batch_does_not_prevent_later_batches():
    provider = _FailsOnCallNumberProvider(fail_on_call_number=2)
    chunks = [_make_chunk(symbol_name=f"chunk_{i}") for i in range(6)]

    result = embed_chunks(chunks, provider, repository_id="repo-1", batch_size=2)

    assert provider.call_count == 3  # all three batches were attempted
    assert len(result.failures) == 2  # batch 2 (the failing one)
    assert len(result.embedded) == 4  # batches 1 and 3



# 18. Partial success across batches


def test_partial_success_across_batches_populates_both_lists_correctly():
    provider = _FailsOnCallNumberProvider(fail_on_call_number=2)
    chunks = [
        _make_chunk(symbol_name="a", file_path="a.py"),
        _make_chunk(symbol_name="b", file_path="b.py"),
        _make_chunk(symbol_name="c", file_path="c.py"),
        _make_chunk(symbol_name="d", file_path="d.py"),
    ]

    result = embed_chunks(chunks, provider, repository_id="repo-1", batch_size=2)

    assert [c.symbol_name for c in result.embedded] == ["a", "b"]
    assert [f.symbol_name for f in result.failures] == ["c", "d"]



# 19. Every batch fails


def test_every_batch_failing_returns_all_chunks_as_failures_without_raising():
    provider = _RaisingProvider()
    chunks = [_make_chunk(symbol_name=f"chunk_{i}") for i in range(5)]

    result = embed_chunks(chunks, provider, repository_id="repo-1", batch_size=2)

    assert result.embedded == []
    assert len(result.failures) == 5
    assert {f.symbol_name for f in result.failures} == {f"chunk_{i}" for i in range(5)}



# 20. Empty string content


def test_chunk_with_empty_string_content_is_still_sent_to_provider():
    provider = FakeEmbeddingProvider()
    chunk = _make_chunk(content="")

    result = embed_chunks([chunk], provider, repository_id="repo-1")

    assert provider.received_batches == [[""]]
    assert len(result.embedded) == 1
    assert result.embedded[0].content == ""



# 21. Duplicate chunks


def test_duplicate_chunks_are_not_deduplicated():
    provider = FakeEmbeddingProvider()
    chunk = _make_chunk()
    duplicate = _make_chunk()  # identical identity fields and content

    result = embed_chunks([chunk, duplicate], provider, repository_id="repo-1")

    assert len(result.embedded) == 2
    assert provider.received_batches == [[chunk.content, duplicate.content]]
    # Both entries are processed independently and, since their identity
    # fields are identical, they legitimately share the same chunk_id -
    # this service has no deduplication logic to collapse them.
    assert result.embedded[0].chunk_id == result.embedded[1].chunk_id