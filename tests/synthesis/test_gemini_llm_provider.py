from types import SimpleNamespace

import pytest

from app.synthesis.gemini_llm_provider import GeminiLLMProvider


class FakeModels:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeClient:
    def __init__(self, response=None):
        self.models = FakeModels(response)


def test_generate_answer_uses_model_and_includes_question_and_context():
    client = FakeClient(SimpleNamespace(text="The function is in module.py."))
    provider = GeminiLLMProvider(client=client)

    answer = provider.generate_answer(
        "Where is greet defined?",
        "File: module.py\nFunction: greet",
    )

    assert answer == "The function is in module.py."
    call = client.models.calls[0]
    assert call["model"] == "gemini-flash-lite-latest"
    assert "Where is greet defined?" in call["contents"]
    assert "File: module.py" in call["contents"]
    assert "only the repository context" in call["contents"]


def test_generate_answer_supports_custom_model():
    client = FakeClient(SimpleNamespace(text="Answer"))
    provider = GeminiLLMProvider(model="custom-model", client=client)

    provider.generate_answer("question", "context")

    assert client.models.calls[0]["model"] == "custom-model"


def test_generate_answer_rejects_empty_text_response():
    provider = GeminiLLMProvider(client=FakeClient(SimpleNamespace(text=None)))

    with pytest.raises(ValueError, match="empty text response"):
        provider.generate_answer("question", "context")


def test_generate_answer_propagates_api_errors():
    provider = GeminiLLMProvider(
        client=FakeClient(RuntimeError("API unavailable"))
    )

    with pytest.raises(RuntimeError, match="API unavailable"):
        provider.generate_answer("question", "context")
