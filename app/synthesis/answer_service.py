from __future__ import annotations

from app.pipeline.repository_query_service import RepositoryQueryService
from app.synthesis.llm_provider import LLMProvider
from app.synthesis.models import AnswerResult
from app.synthesis.prompt import build_context

NO_CONTEXT_ANSWER = "No relevant repository context was found for this question."


class AnswerService:

    def __init__(self, query_service: RepositoryQueryService, llm_provider: LLMProvider) -> None:
        self._query_service = query_service
        self._llm_provider = llm_provider

    def answer(self, repository_id: str, question: str, top_k: int = 5) -> AnswerResult:
        chunks = self._query_service.ask(repository_id, question, top_k)

        if not chunks:
            return AnswerResult(answer=NO_CONTEXT_ANSWER, sources=[])

        context = build_context(chunks)
        generated_answer = self._llm_provider.generate_answer(question, context)
        return AnswerResult(answer=generated_answer, sources=chunks)