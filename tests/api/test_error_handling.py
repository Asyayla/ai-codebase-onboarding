from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.api.app import create_app


def _gemini_error(code: int, status: str) -> genai_errors.APIError:
    payload = {"error": {"code": code, "message": "failure", "status": status}}
    if code >= 500 or code == 429:
        return genai_errors.ServerError(code, payload)
    return genai_errors.ClientError(code, payload)


class _FailingAnswerService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def answer(self, repository_id, question, top_k):
        raise self._error


class _FailingQueryService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def index_repository(self, source):
        raise self._error


def _ask(error: Exception):
    client = TestClient(create_app(answer_service=_FailingAnswerService(error)))
    return client.post(
        "/repositories/https%3A%2F%2Fgithub.com%2Fexample%2Frepo/ask",
        json={"question": "What does this project do?"},
    )


def test_ask_returns_503_when_gemini_is_temporarily_unavailable():
    response = _ask(_gemini_error(503, "UNAVAILABLE"))

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


def test_ask_returns_503_when_gemini_rate_limits_the_request():
    response = _ask(_gemini_error(429, "RESOURCE_EXHAUSTED"))

    assert response.status_code == 503


def test_ask_returns_502_when_gemini_rejects_the_request():
    response = _ask(_gemini_error(400, "INVALID_ARGUMENT"))

    assert response.status_code == 502
    assert "rejected" in response.json()["detail"]


def test_index_returns_503_when_gemini_embedding_service_is_unavailable():
    client = TestClient(
        create_app(
            query_service=_FailingQueryService(_gemini_error(503, "UNAVAILABLE"))
        )
    )

    response = client.post(
        "/repositories/index", json={"source": "https://github.com/example/repo"}
    )

    assert response.status_code == 503