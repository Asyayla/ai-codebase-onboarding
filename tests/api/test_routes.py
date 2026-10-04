from __future__ import annotations

from typing import List, Tuple

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.persistence.models import StoredChunk
from app.pipeline.models import RepositoryIndexingResult
from app.retrieval.models import RetrievedChunk
from app.synthesis.models import AnswerResult


class _FakeQueryService:
    def __init__(self, result_to_return: RepositoryIndexingResult) -> None:
        self._result_to_return = result_to_return
        self.calls: List[str] = []

    def index_repository(self, source: str) -> RepositoryIndexingResult:
        self.calls.append(source)
        return self._result_to_return

    def ask(self, repository_id, query, top_k=5):
        raise AssertionError("ask() should not be called through the index endpoint")


class _FakeAnswerService:
    def __init__(self, result_to_return: AnswerResult) -> None:
        self._result_to_return = result_to_return
        self.calls: List[Tuple[str, str, int]] = []

    def answer(self, repository_id: str, question: str, top_k: int = 5) -> AnswerResult:
        self.calls.append((repository_id, question, top_k))
        return self._result_to_return


def _make_indexing_result(**overrides) -> RepositoryIndexingResult:
    defaults = dict(
        repository_id="repo-1",
        files_considered=10,
        file_outcomes=[],
        embedded=[],
        embedding_failures=[],
        new_chunk_ids=["a", "b"],
        changed_chunk_ids=["c"],
        unchanged_chunk_ids=["d", "e", "f"],
        removed_chunk_ids=[],
    )
    defaults.update(overrides)
    return RepositoryIndexingResult(**defaults)


def _make_retrieved_chunk(chunk_id: str, file_path: str, score: float) -> RetrievedChunk:
    stored = StoredChunk(
        chunk_id=chunk_id,
        repository_id="repo-1",
        file_path=file_path,
        language="python",
        symbol_type="function",
        symbol_name="foo",
        start_line=1,
        end_line=10,
        content="def foo(): pass",
        parent_symbol_name=None,
        docstring=None,
        embedding=[0.1, 0.2, 0.3],
        embedding_model="text-embedding-3-small",
        content_hash="deadbeef",
    )
    return RetrievedChunk(stored_chunk=stored, score=score, rank=999)


def _client_with(query_service=None, answer_service=None) -> TestClient:
    app = create_app(query_service=query_service, answer_service=answer_service)
    return TestClient(app)



def test_index_endpoint_calls_query_service_with_the_supplied_source():
    fake_query_service = _FakeQueryService(_make_indexing_result())
    client = _client_with(query_service=fake_query_service)

    client.post("/repositories/index", json={"source": "https://github.com/example/repo"})

    assert fake_query_service.calls == ["https://github.com/example/repo"]


def test_index_endpoint_returns_api_safe_response():
    fake_query_service = _FakeQueryService(_make_indexing_result())
    client = _client_with(query_service=fake_query_service)

    response = client.post("/repositories/index", json={"source": "https://github.com/example/repo"})

    assert response.status_code == 200
    assert response.json() == {
        "repository_id": "repo-1",
        "files_considered": 10,
        "new_chunk_count": 2,
        "changed_chunk_count": 1,
        "unchanged_chunk_count": 3,
        "removed_chunk_count": 0,
        "embedding_failure_count": 0,
    }


def test_index_endpoint_rejects_empty_source():
    fake_query_service = _FakeQueryService(_make_indexing_result())
    client = _client_with(query_service=fake_query_service)

    response = client.post("/repositories/index", json={"source": ""})

    assert response.status_code == 422


def test_index_endpoint_rejects_whitespace_only_source():
    fake_query_service = _FakeQueryService(_make_indexing_result())
    client = _client_with(query_service=fake_query_service)

    response = client.post("/repositories/index", json={"source": "   "})

    assert response.status_code == 422




def test_ask_endpoint_calls_answer_service_with_repository_id_question_and_top_k():
    chunks = [_make_retrieved_chunk("c1", "app/foo.py", 0.9)]
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=chunks))
    client = _client_with(answer_service=fake_answer_service)

    client.post("/repositories/repo-42/ask", json={"question": "What does foo do?", "top_k": 3})

    assert fake_answer_service.calls == [("repo-42", "What does foo do?", 3)]


def test_ask_endpoint_uses_default_top_k_of_5():
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=[]))
    client = _client_with(answer_service=fake_answer_service)

    client.post("/repositories/repo-42/ask", json={"question": "What does foo do?"})

    assert fake_answer_service.calls[0][2] == 5


def test_ask_endpoint_returns_answer_and_sources():
    chunks = [_make_retrieved_chunk("c1", "app/foo.py", 0.9)]
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=chunks))
    client = _client_with(answer_service=fake_answer_service)

    response = client.post("/repositories/repo-42/ask", json={"question": "What does foo do?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "the answer"
    assert body["sources"] == [
        {"file_path": "app/foo.py", "chunk_id": "c1", "score": 0.9, "rank": 1}
    ]


def test_ask_endpoint_assigns_source_ranks_starting_at_one_in_order():
    chunks = [
        _make_retrieved_chunk("first", "a.py", 0.9),
        _make_retrieved_chunk("second", "b.py", 0.8),
        _make_retrieved_chunk("third", "c.py", 0.7),
    ]
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=chunks))
    client = _client_with(answer_service=fake_answer_service)

    response = client.post("/repositories/repo-42/ask", json={"question": "question"})

    ranks = [source["rank"] for source in response.json()["sources"]]
    assert ranks == [1, 2, 3]


def test_ask_endpoint_does_not_trust_the_internal_rank_value():
    chunks = [_make_retrieved_chunk("c1", "app/foo.py", 0.9)]
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=chunks))
    client = _client_with(answer_service=fake_answer_service)

    response = client.post("/repositories/repo-42/ask", json={"question": "question"})

    assert response.json()["sources"][0]["rank"] == 1


def test_ask_endpoint_rejects_empty_question():
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=[]))
    client = _client_with(answer_service=fake_answer_service)

    response = client.post("/repositories/repo-42/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_endpoint_rejects_whitespace_only_question():
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=[]))
    client = _client_with(answer_service=fake_answer_service)

    response = client.post("/repositories/repo-42/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_endpoint_rejects_zero_top_k():
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=[]))
    client = _client_with(answer_service=fake_answer_service)

    response = client.post("/repositories/repo-42/ask", json={"question": "question", "top_k": 0})

    assert response.status_code == 422


def test_ask_endpoint_rejects_negative_top_k():
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=[]))
    client = _client_with(answer_service=fake_answer_service)

    response = client.post("/repositories/repo-42/ask", json={"question": "question", "top_k": -1})

    assert response.status_code == 422




def test_routes_work_with_injected_fakes_without_openai_api_key_or_database_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    fake_query_service = _FakeQueryService(_make_indexing_result())
    fake_answer_service = _FakeAnswerService(AnswerResult(answer="the answer", sources=[]))
    client = _client_with(query_service=fake_query_service, answer_service=fake_answer_service)

    index_response = client.post(
        "/repositories/index", json={"source": "https://github.com/example/repo"}
    )
    ask_response = client.post("/repositories/repo-1/ask", json={"question": "question"})

    assert index_response.status_code == 200
    assert ask_response.status_code == 200
