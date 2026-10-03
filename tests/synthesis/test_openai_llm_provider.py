from __future__ import annotations

from typing import Any, List, Optional

import pytest
from openai import OpenAIError

from app.synthesis.openai_llm_provider import OpenAILLMProvider


class _FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _FakeResponsesResource:
    """Stands in for client.responses, recording calls made to .create()."""

    def __init__(
        self,
        response: Optional[_FakeResponse] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: List[dict] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _FakeOpenAIClient:
    def __init__(self, responses: _FakeResponsesResource) -> None:
        self.responses = responses




def test_default_model_is_gpt_6_luna():
    responses_resource = _FakeResponsesResource(response=_FakeResponse("answer"))
    provider = OpenAILLMProvider(client=_FakeOpenAIClient(responses_resource))

    provider.generate_answer("question", "context")

    assert responses_resource.calls[0]["model"] == "gpt-6-luna"


def test_model_is_overridable_via_constructor():
    responses_resource = _FakeResponsesResource(response=_FakeResponse("answer"))
    provider = OpenAILLMProvider(model="gpt-custom", client=_FakeOpenAIClient(responses_resource))

    provider.generate_answer("question", "context")

    assert responses_resource.calls[0]["model"] == "gpt-custom"




def test_generate_answer_calls_responses_create_exactly_once():
    responses_resource = _FakeResponsesResource(response=_FakeResponse("answer"))
    provider = OpenAILLMProvider(client=_FakeOpenAIClient(responses_resource))

    provider.generate_answer("question", "context")

    assert len(responses_resource.calls) == 1




def test_generate_answer_includes_the_question_in_the_request():
    responses_resource = _FakeResponsesResource(response=_FakeResponse("answer"))
    provider = OpenAILLMProvider(client=_FakeOpenAIClient(responses_resource))

    provider.generate_answer("What does foo() do?", "some context")

    assert "What does foo() do?" in responses_resource.calls[0]["input"]


def test_generate_answer_includes_the_context_in_the_request():
    responses_resource = _FakeResponsesResource(response=_FakeResponse("answer"))
    provider = OpenAILLMProvider(client=_FakeOpenAIClient(responses_resource))

    provider.generate_answer("question", "some repository context here")

    assert "some repository context here" in responses_resource.calls[0]["input"]


def test_generate_answer_sends_non_empty_grounding_instructions():
    responses_resource = _FakeResponsesResource(response=_FakeResponse("answer"))
    provider = OpenAILLMProvider(client=_FakeOpenAIClient(responses_resource))

    provider.generate_answer("question", "context")

    assert responses_resource.calls[0]["instructions"]




def test_generate_answer_returns_output_text_from_response():
    responses_resource = _FakeResponsesResource(response=_FakeResponse("the generated answer"))
    provider = OpenAILLMProvider(client=_FakeOpenAIClient(responses_resource))

    result = provider.generate_answer("question", "context")

    assert result == "the generated answer"




def test_generate_answer_uses_the_injected_client_instance():
    responses_resource = _FakeResponsesResource(response=_FakeResponse("answer"))
    injected_client = _FakeOpenAIClient(responses_resource)
    provider = OpenAILLMProvider(client=injected_client)

    provider.generate_answer("question", "context")

    assert len(responses_resource.calls) == 1




def test_generate_answer_propagates_api_exceptions():
    class _SimulatedApiError(Exception):
        pass

    responses_resource = _FakeResponsesResource(error=_SimulatedApiError("boom"))
    provider = OpenAILLMProvider(client=_FakeOpenAIClient(responses_resource))

    with pytest.raises(_SimulatedApiError):
        provider.generate_answer("question", "context")




def test_default_client_raises_openai_error_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIError, match="api_key"):
        OpenAILLMProvider()