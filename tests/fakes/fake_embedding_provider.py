"""
fake_embedding_provider.py

An in-memory, offline stand-in for a real EmbeddingProvider, for use in
tests. Structurally satisfies app.embedding.provider.EmbeddingProvider
(no inheritance needed - Protocol conformance is structural) with zero
network calls, zero disk I/O, and zero external dependencies.

Vectors are derived deterministically from each input text via SHA-256, so
the same text always produces the same vector - in this call, across
calls, across separate provider instances, and across separate test runs
and processes. This mirrors the same deterministic-hash approach already
used for chunk_id in app/embedding/chunk_identity.py, rather than
introducing a different pattern for "stable, reproducible identifiers"
within the same layer of the codebase.
"""

from __future__ import annotations

import hashlib
from typing import List


def _generate_vector(text: str, dimensions: int) -> List[float]:
    """Deterministically derive a `dimensions`-length vector of floats in
    [0, 1] from `text`, using SHA-256.

    A single SHA-256 digest is 32 bytes, which may be fewer than
    `dimensions` requires, so additional digest blocks are generated (by
    hashing the text combined with a block index) and concatenated until
    enough bytes are available. Each byte (0-255) is scaled to [0, 1] by
    dividing by 255. This is not a cryptographic use of SHA-256 - it is
    simply a convenient, stable source of deterministic pseudo-random
    bytes with no dependency on Python's process-randomized hash().
    """
    blocks: List[bytes] = []
    total_bytes = 0
    block_index = 0
    while total_bytes < dimensions:
        block = hashlib.sha256(f"{text}:{block_index}".encode("utf-8")).digest()
        blocks.append(block)
        total_bytes += len(block)
        block_index += 1

    raw_bytes = b"".join(blocks)[:dimensions]
    return [byte / 255.0 for byte in raw_bytes]


class FakeEmbeddingProvider:
    """Offline EmbeddingProvider for tests.

    Structurally satisfies app.embedding.provider.EmbeddingProvider:
    embed_batch(), model_name, and dimensions all match. There is no
    import-time or inheritance-time dependency on the Protocol - any code
    that type-checks or calls this the same way it would call a real
    provider is exercising real Protocol conformance.
    """

    def __init__(self, dimensions: int = 8, model_name: str = "fake-embedding-model"):
        if dimensions <= 0:
            raise ValueError(f"dimensions must be a positive integer, got {dimensions}")

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
        """Every batch of texts this provider has been asked to embed, in
        call order. Test-inspection state only - not part of the
        EmbeddingProvider Protocol itself."""
        return self._received_batches

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Store a copy, not the caller's original list object, so that if
        # the caller mutates their list after this call, recorded history
        # here is unaffected.
        self._received_batches.append(list(texts))

        return [_generate_vector(text, self._dimensions) for text in texts]