from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

_MODEL_NAME = "text-embedding-3-small"
_DIMENSIONS = 1536


class OpenAIEmbeddingProvider:
    """Production EmbeddingProvider backed by the OpenAI embeddings API.

    Reads OPENAI_API_KEY from the environment via the standard OpenAI()
    client behavior; no key handling is done here.
    """

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        self._client = client if client is not None else OpenAI()

    @property
    def model_name(self) -> str:
        return _MODEL_NAME

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=_MODEL_NAME,
            input=texts,
            encoding_format="float",
        )

        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
