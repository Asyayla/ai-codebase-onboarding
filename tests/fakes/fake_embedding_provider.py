from __future__ import annotations

import hashlib
from typing import List


def _generate_vector(text: str, dimensions: int) -> List[float]:
    """Generate a deterministic vector from text using SHA-256."""
    blocks: List[bytes] = []
    total_bytes = 0
    block_index = 0

    while total_bytes < dimensions:
        block = hashlib.sha256(
            f"{text}:{block_index}".encode("utf-8")
        ).digest()
        blocks.append(block)
        total_bytes += len(block)
        block_index += 1

    raw_bytes = b"".join(blocks)[:dimensions]
    return [byte / 255.0 for byte in raw_bytes]


class FakeEmbeddingProvider:
    """Offline EmbeddingProvider implementation for tests."""

    def __init__(
        self,
        dimensions: int = 8,
        model_name: str = "fake-embedding-model",
    ):
        if dimensions <= 0:
            raise ValueError(
                f"dimensions must be a positive integer, got {dimensions}"
            )

        self._dimensions = dimensions
        self._model_name = model_name
        self._received_batches: List[List[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def received_batches(self) -> List[List[str]]:
        """Return batches received for test inspection."""
        return self._received_batches

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Store a copy so later mutations by the caller do not affect history.
        self._received_batches.append(list(texts))

        return [_generate_vector(text, self._dimensions) for text in texts]