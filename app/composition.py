from __future__ import annotations

from typing import Optional

from app.embedding.provider import EmbeddingProvider
from app.embedding.providers.openai_provider import OpenAIEmbeddingProvider
from app.persistence.chunk_store import ChunkStore
from app.persistence.postgres_chunk_store import PostgresChunkStore
from app.pipeline.repository_query_service import RepositoryQueryService
from app.retrieval.postgres_vector_store import PostgresVectorStore
from app.retrieval.vector_store import VectorStore

DEFAULT_BATCH_SIZE = 16


def build_repository_query_service(
    *,
    embedding_provider: Optional[EmbeddingProvider] = None,
    chunk_store: Optional[ChunkStore] = None,
    vector_store: Optional[VectorStore] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RepositoryQueryService:
    """Compose the production RepositoryQueryService from real implementations.

    Each dependency defaults to its production implementation - reading
    configuration (OPENAI_API_KEY, DATABASE_URL) from the environment only
    when it is actually constructed, never at import time - and can be
    overridden, e.g. with fakes in tests or a future alternative provider.
    """
    if embedding_provider is None:
        embedding_provider = OpenAIEmbeddingProvider()
    if chunk_store is None:
        chunk_store = PostgresChunkStore()
    if vector_store is None:
        vector_store = PostgresVectorStore()

    return RepositoryQueryService(
        embedding_provider,
        chunk_store,
        vector_store,
        batch_size,
    )