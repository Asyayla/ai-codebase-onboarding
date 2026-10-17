from __future__ import annotations

from typing import List, Optional

from google import genai
from google.genai import types

_MODEL_NAME = "gemini-embedding-001"
_DIMENSIONS = 1536


class GeminiEmbeddingProvider:
    """Embedding provider backed by the Gemini embedding API."""

    def __init__(self, client=None) -> None:
        self._client = client if client is not None else genai.Client()

    @property
    def model_name(self) -> str:
        return _MODEL_NAME

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed repository chunks for indexing."""
        if not texts:
            return []

        response = self._client.models.embed_content(
            model=_MODEL_NAME,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=_DIMENSIONS,
            ),
        )

        vectors = [embedding.values for embedding in response.embeddings]
        if len(vectors) != len(texts):
            raise ValueError(
                f"expected {len(texts)} embedding vectors, got {len(vectors)}"
            )
        return vectors

    def embed_query(self, query: str) -> List[float]:
        """Embed a natural-language query for code retrieval."""
        response = self._client.models.embed_content(
            model=_MODEL_NAME,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="CODE_RETRIEVAL_QUERY",
                output_dimensionality=_DIMENSIONS,
            ),
        )

        embeddings = response.embeddings or []
        if len(embeddings) != 1:
            raise ValueError(
                f"expected exactly one embedding vector for one query, "
                f"got {len(embeddings)}"
            )

        vector = embeddings[0].values
        if len(vector) != _DIMENSIONS:
            raise ValueError(
                f"query embedding has {len(vector)} dimensions, "
                f"expected {_DIMENSIONS}"
            )
        return vector
