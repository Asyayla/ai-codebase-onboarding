from __future__ import annotations

from typing import Optional

from openai import OpenAI

_DEFAULT_MODEL = "gpt-6-luna"

_GROUNDING_INSTRUCTIONS = (
    "You are a codebase onboarding assistant. Answer the question using "
    "only the repository context provided below. Do not invent facts that "
    "are not supported by that context. If the context is insufficient to "
    "answer the question, say so explicitly rather than guessing."
)


class OpenAILLMProvider:

    def __init__(self, model: str = _DEFAULT_MODEL, client: Optional[OpenAI] = None) -> None:
        self._model = model
        self._client = client if client is not None else OpenAI()

    def generate_answer(self, question: str, context: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            instructions=_GROUNDING_INSTRUCTIONS,
            input=f"Repository context:\n{context}\n\nQuestion: {question}",
        )
        return response.output_text