from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class IndexRequest(BaseModel):
    source: str

    @field_validator("source")
    @classmethod
    def source_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source must not be empty or whitespace-only")
        return value


class IndexResponse(BaseModel):
    repository_id: str
    files_considered: int
    new_chunk_count: int
    changed_chunk_count: int
    unchanged_chunk_count: int
    removed_chunk_count: int
    embedding_failure_count: int


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, gt=0, le=50)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return value


class SourceResponse(BaseModel):
    file_path: str
    chunk_id: str
    score: float
    rank: int


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]
