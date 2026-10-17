from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.models import (
    AskRequest,
    AskResponse,
    IndexRequest,
    IndexResponse,
    SourceResponse,
)
from app.pipeline.repository_query_service import RepositoryQueryService
from app.synthesis.answer_service import AnswerService

router = APIRouter()


def get_query_service() -> RepositoryQueryService:
    raise RuntimeError(
        "get_query_service was not overridden; create the app via app.api.app.create_app()"
    )


def get_answer_service() -> AnswerService:
    raise RuntimeError(
        "get_answer_service was not overridden; create the app via app.api.app.create_app()"
    )


@router.post("/repositories/index", response_model=IndexResponse)
def index_repository(
    request: IndexRequest,
    query_service: RepositoryQueryService = Depends(get_query_service),
) -> IndexResponse:
    result = query_service.index_repository(request.source)
    return IndexResponse(
        repository_id=result.repository_id,
        files_considered=result.files_considered,
        new_chunk_count=len(result.new_chunk_ids),
        changed_chunk_count=len(result.changed_chunk_ids),
        unchanged_chunk_count=len(result.unchanged_chunk_ids),
        removed_chunk_count=len(result.removed_chunk_ids),
        embedding_failure_count=len(result.embedding_failures),
    )


@router.post("/repositories/{repository_id:path}/ask", response_model=AskResponse)
def ask_repository(
    repository_id: str,
    request: AskRequest,
    answer_service: AnswerService = Depends(get_answer_service),
) -> AskResponse:
    result = answer_service.answer(repository_id, request.question, request.top_k)
    sources = [
        SourceResponse(
            file_path=chunk.stored_chunk.file_path,
            chunk_id=chunk.stored_chunk.chunk_id,
            score=chunk.score,
            rank=rank,
        )
        for rank, chunk in enumerate(result.sources, start=1)
    ]
    return AskResponse(answer=result.answer, sources=sources)