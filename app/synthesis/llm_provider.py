from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):

    def generate_answer(self, question: str, context: str) -> str:
        ...