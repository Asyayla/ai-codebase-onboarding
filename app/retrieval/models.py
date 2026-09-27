from __future__ import annotations

from dataclasses import dataclass

from app.persistence.models import StoredChunk


@dataclass
class RetrievedChunk:
    """A persisted chunk returned by a similarity search."""

    stored_chunk: StoredChunk
    score: float
    rank: int