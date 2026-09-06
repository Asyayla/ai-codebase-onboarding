"""
provider.py

Defines the EmbeddingProvider contract that any embedding backend must
satisfy - real or fake. Nothing in this file knows about any specific
provider (no Voyage, no OpenAI, no HTTP, no API keys); it exists purely so
the rest of the application can depend on a shape, not an implementation.

EmbeddingProvider is a structural (Protocol) type, not an abstract base
class - a class satisfies it simply by having matching methods/properties,
with no inheritance required. This keeps concrete providers (real or fake)
fully decoupled from this module at import time.
"""

from __future__ import annotations

from typing import List, Protocol


class EmbeddingProvider(Protocol):
    """Contract for anything that can turn text into embedding vectors.

    Implementations are responsible for whatever it takes to produce
    vectors (an HTTP call to a real API, or an in-memory computation for a
    fake/test provider) - this Protocol only constrains behavior, not how
    an implementation is constructed or configured.
    """

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.

        Contract:
            - Returns exactly one output vector per input text.
            - Output order corresponds to input order: result[i] is the
              embedding for texts[i].
            - Empty input (texts == []) returns an empty list ([]), never
              an error.
            - Every returned vector has exactly `self.dimensions` values.

        Args:
            texts: the texts to embed. May be empty.

        Returns:
            One embedding vector per input text, in the same order.
        """
        ...

    @property
    def model_name(self) -> str:
        """The identifier of the model producing these embeddings (e.g.
        used to stamp EmbeddedChunk.embedding_model)."""
        ...

    @property
    def dimensions(self) -> int:
        """The length of every vector returned by embed_batch."""
        ...