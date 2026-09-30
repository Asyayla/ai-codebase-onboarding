from __future__ import annotations

from typing import List

from app.embedding.provider import EmbeddingProvider
from app.persistence.chunk_store import ChunkStore
from app.pipeline.models import RepositoryIndexingResult
from app.pipeline.repository_indexer import RepositoryIndexer
from app.retrieval.models import RetrievedChunk
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


class RepositoryQueryService:
    """Thin composition of RepositoryIndexer and Retriever sharing one
    EmbeddingProvider instance."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunk_store: ChunkStore,
        vector_store: VectorStore,
        batch_size: int = 16,
    ) -> None:
        self._indexer = RepositoryIndexer(
            embedding_provider,
            chunk_store,
            batch_size,
        )
        self._retriever = Retriever(
            embedding_provider,
            vector_store,
        )

    def index_repository(self, source: str) -> RepositoryIndexingResult:
        return self._indexer.index_repository(source)

    def ask(
        self,
        repository_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        return self._retriever.retrieve(repository_id, query, top_k)