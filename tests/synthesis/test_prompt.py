from __future__ import annotations

from app.persistence.models import StoredChunk
from app.retrieval.models import RetrievedChunk
from app.synthesis.prompt import build_context


def _make_stored_chunk(chunk_id: str, file_path: str, content: str) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        repository_id="repo-1",
        file_path=file_path,
        language="python",
        symbol_type="function",
        symbol_name="some_function",
        start_line=1,
        end_line=10,
        content=content,
        parent_symbol_name=None,
        docstring=None,
        embedding=[0.1, 0.2, 0.3],
        embedding_model="text-embedding-3-small",
        content_hash="deadbeef",
    )


def _make_retrieved_chunk(
    chunk_id: str, file_path: str, content: str, score: float, rank: int
) -> RetrievedChunk:
    return RetrievedChunk(
        stored_chunk=_make_stored_chunk(chunk_id, file_path, content),
        score=score,
        rank=rank,
    )




def test_build_context_with_empty_list_returns_empty_string():
    assert build_context([]) == ""




def test_build_context_includes_file_path_for_each_chunk():
    chunks = [_make_retrieved_chunk("c1", "app/foo.py", "def foo(): pass", 0.9, 1)]

    context = build_context(chunks)

    assert "app/foo.py" in context


def test_build_context_includes_chunk_id_for_each_chunk():
    chunks = [_make_retrieved_chunk("c1", "app/foo.py", "def foo(): pass", 0.9, 1)]

    context = build_context(chunks)

    assert "c1" in context


def test_build_context_includes_raw_content_for_each_chunk():
    chunks = [_make_retrieved_chunk("c1", "app/foo.py", "def foo(): pass", 0.9, 1)]

    context = build_context(chunks)

    assert "def foo(): pass" in context


def test_build_context_includes_score_for_each_chunk():
    chunks = [_make_retrieved_chunk("c1", "app/foo.py", "def foo(): pass", 0.9123, 1)]

    context = build_context(chunks)

    assert "0.9123" in context



def test_build_context_preserves_input_order():
    chunks = [
        _make_retrieved_chunk("first", "a.py", "AAA", 0.5, 1),
        _make_retrieved_chunk("second", "b.py", "BBB", 0.4, 2),
    ]

    context = build_context(chunks)

    assert context.index("AAA") < context.index("BBB")


def test_build_context_separates_multiple_chunks_distinguishably():
    chunks = [
        _make_retrieved_chunk("first", "a.py", "AAA", 0.5, 1),
        _make_retrieved_chunk("second", "b.py", "BBB", 0.4, 2),
    ]

    context = build_context(chunks)

    # Each chunk's content must appear as its own section, never run
    # together with the next chunk's content with no boundary between them.
    assert "AAABBB" not in context




def test_build_context_is_deterministic_for_the_same_input():
    chunks = [_make_retrieved_chunk("c1", "app/foo.py", "def foo(): pass", 0.9, 1)]

    first = build_context(chunks)
    second = build_context(chunks)

    assert first == second


def test_build_context_does_not_mutate_input_list():
    chunks = [
        _make_retrieved_chunk("first", "a.py", "AAA", 0.5, 1),
        _make_retrieved_chunk("second", "b.py", "BBB", 0.4, 2),
    ]
    original = list(chunks)

    build_context(chunks)

    assert chunks == original