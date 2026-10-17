from __future__ import annotations

from google import genai

_DEFAULT_MODEL = "gemini-flash-lite-latest"

_GROUNDING_INSTRUCTIONS = (
    "You are a codebase onboarding assistant. Answer the question using "
    "only the repository context provided below. Do not invent facts that "
    "are not supported by that context. If the context is insufficient to "
    "answer the question, say so explicitly rather than guessing."
)


class GeminiLLMProvider:
    """Answer-generation provider backed by the Gemini API."""

    def __init__(self, model: str = _DEFAULT_MODEL, client=None) -> None:
        self._model = model
        self._client = client if client is not None else genai.Client()

    def generate_answer(self, question: str, context: str) -> str:
        prompt = (
            f"{_GROUNDING_INSTRUCTIONS}\n\n"
            f"Repository context:\n{context}\n\n"
            f"Question: {question}"
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        answer = response.text
        if answer is None:
            raise ValueError("Gemini returned an empty text response")
        return answer