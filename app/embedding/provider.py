from __future__ import annotations

from typing import List, Protocol


class EmbeddingProvider(Protocol):
    """Contract for embedding providers."""

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts in input order, returning one vector per text."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model."""
        ...

    @property
    def dimensions(self) -> int:
        """Dimension of each embedding vector."""
        ...