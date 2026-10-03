from __future__ import annotations

from typing import List, Tuple


class FakeLLMProvider:

    def __init__(self, answer: str = "fake answer") -> None:
        self._answer = answer
        self.calls: List[Tuple[str, str]] = []

    def generate_answer(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        return self._answer