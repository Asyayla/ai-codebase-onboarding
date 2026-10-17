from __future__ import annotations

from typing import Optional

from app.embedding.provider import EmbeddingProvider
from app.embedding.providers.gemini_provider import GeminiEmbeddingProvider
from app.persistence.chunk_store import ChunkStore
from app.persistence.postgres_chunk_store import PostgresChunkStore
from app.pipeline.repository_query_service import RepositoryQueryService
from app.retrieval.postgres_vector_store import PostgresVectorStore
from app.retrieval.vector_store import VectorStore
from app.synthesis.answer_service import AnswerService
from app.synthesis.llm_provider import LLMProvider
from app.synthesis.gemini_llm_provider import GeminiLLMProvider

DEFAULT_BATCH_SIZE = 16


def build_repository_query_service(
    *,
    embedding_provider: Optional[EmbeddingProvider] = None,
    chunk_store: Optional[ChunkStore] = None,
    vector_store: Optional[VectorStore] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RepositoryQueryService:
    if embedding_provider is None:
        embedding_provider = GeminiEmbeddingProvider()
    if chunk_store is None:
        chunk_store = PostgresChunkStore()
    if vector_store is None:
        vector_store = PostgresVectorStore()
    return RepositoryQueryService(embedding_provider, chunk_store, vector_store, batch_size)


def build_answer_service(
    *,
    query_service: Optional[RepositoryQueryService] = None,
    llm_provider: Optional[LLMProvider] = None,
) -> AnswerService:
    if query_service is None:
        query_service = build_repository_query_service()
    if llm_provider is None:
        llm_provider = GeminiLLMProvider()
    return AnswerService(query_service, llm_provider)