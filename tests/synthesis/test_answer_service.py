from __future__ import annotations

from typing import List, Tuple

from app.persistence.models import StoredChunk
from app.retrieval.models import RetrievedChunk
from app.synthesis.answer_service import NO_CONTEXT_ANSWER, AnswerService
from app.synthesis.prompt import build_context
from tests.fakes.fake_llm_provider import FakeLLMProvider


class _FakeQueryService:

    def __init__(self, chunks_to_return: List[RetrievedChunk]) -> None:
        self._chunks_to_return = chunks_to_return
        self.calls: List[Tuple[str, str, int]] = []

    def ask(self, repository_id: str, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        self.calls.append((repository_id, query, top_k))
        return self._chunks_to_return


def _make_retrieved_chunk(chunk_id: str, content: str, score: float, rank: int) -> RetrievedChunk:
    stored = StoredChunk(
        chunk_id=chunk_id,
        repository_id="repo-1",
        file_path="app/foo.py",
        language="python",
        symbol_type="function",
        symbol_name="foo",
        start_line=1,
        end_line=10,
        content=content,
        parent_symbol_name=None,
        docstring=None,
        embedding=[0.1, 0.2, 0.3],
        embedding_model="text-embedding-3-small",
        content_hash="deadbeef",
    )
    return RetrievedChunk(stored_chunk=stored, score=score, rank=rank)




def test_answer_returns_fixed_no_context_message_when_no_chunks_retrieved():
    query_service = _FakeQueryService(chunks_to_return=[])
    llm_provider = FakeLLMProvider()
    service = AnswerService(query_service, llm_provider)

    result = service.answer("repo-1", "What does this do?")

    assert result.answer == NO_CONTEXT_ANSWER


def test_answer_returns_empty_sources_when_no_chunks_retrieved():
    query_service = _FakeQueryService(chunks_to_return=[])
    llm_provider = FakeLLMProvider()
    service = AnswerService(query_service, llm_provider)

    result = service.answer("repo-1", "What does this do?")

    assert result.sources == []


def test_answer_does_not_call_llm_when_no_chunks_retrieved():
    query_service = _FakeQueryService(chunks_to_return=[])
    llm_provider = FakeLLMProvider()
    service = AnswerService(query_service, llm_provider)

    service.answer("repo-1", "What does this do?")

    assert llm_provider.calls == []




def test_answer_calls_llm_with_the_question_when_chunks_found():
    chunks = [_make_retrieved_chunk("c1", "def foo(): pass", 0.9, 1)]
    query_service = _FakeQueryService(chunks_to_return=chunks)
    llm_provider = FakeLLMProvider()
    service = AnswerService(query_service, llm_provider)

    service.answer("repo-1", "What does foo do?")

    assert llm_provider.calls[0][0] == "What does foo do?"


def test_answer_calls_llm_with_context_built_from_the_retrieved_chunks():
    chunks = [_make_retrieved_chunk("c1", "def foo(): pass", 0.9, 1)]
    query_service = _FakeQueryService(chunks_to_return=chunks)
    llm_provider = FakeLLMProvider()
    service = AnswerService(query_service, llm_provider)

    service.answer("repo-1", "What does foo do?")

    assert llm_provider.calls[0][1] == build_context(chunks)


def test_answer_result_sources_are_exactly_the_retrieved_chunks():
    chunks = [_make_retrieved_chunk("c1", "def foo(): pass", 0.9, 1)]
    query_service = _FakeQueryService(chunks_to_return=chunks)
    llm_provider = FakeLLMProvider()
    service = AnswerService(query_service, llm_provider)

    result = service.answer("repo-1", "What does foo do?")

    assert result.sources == chunks


def test_answer_returns_the_llm_generated_text():
    chunks = [_make_retrieved_chunk("c1", "def foo(): pass", 0.9, 1)]
    query_service = _FakeQueryService(chunks_to_return=chunks)
    llm_provider = FakeLLMProvider(answer="foo() does nothing.")
    service = AnswerService(query_service, llm_provider)

    result = service.answer("repo-1", "What does foo do?")

    assert result.answer == "foo() does nothing."




def test_answer_passes_repository_id_question_and_top_k_to_query_service():
    query_service = _FakeQueryService(chunks_to_return=[])
    llm_provider = FakeLLMProvider()
    service = AnswerService(query_service, llm_provider)

    service.answer("repo-42", "some question", top_k=3)

    assert query_service.calls == [("repo-42", "some question", 3)]


def test_answer_uses_default_top_k_of_5_when_not_specified():
    query_service = _FakeQueryService(chunks_to_return=[])
    llm_provider = FakeLLMProvider()
    service = AnswerService(query_service, llm_provider)

    service.answer("repo-42", "some question")

    assert query_service.calls[0][2] == 5