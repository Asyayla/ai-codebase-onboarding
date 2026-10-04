from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from app.api.routes import get_answer_service, get_query_service, router
from app.composition import build_answer_service, build_repository_query_service
from app.pipeline.repository_query_service import RepositoryQueryService
from app.synthesis.answer_service import AnswerService


def create_app(
    *,
    query_service: Optional[RepositoryQueryService] = None,
    answer_service: Optional[AnswerService] = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    app.state.query_service = query_service
    app.state.answer_service = answer_service

    def _get_query_service() -> RepositoryQueryService:
        if app.state.query_service is None:
            app.state.query_service = build_repository_query_service()
        return app.state.query_service

    def _get_answer_service() -> AnswerService:
        if app.state.answer_service is None:
            app.state.answer_service = build_answer_service()
        return app.state.answer_service

    app.dependency_overrides[get_query_service] = _get_query_service
    app.dependency_overrides[get_answer_service] = _get_answer_service

    return app
