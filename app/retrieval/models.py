"""
models.py

Retrieval-specific result model. Defined separately from
app/persistence/models.py so a retrieved result carries retrieval-only
concerns (similarity score, rank) without those concerns leaking into
StoredChunk, which is a persistence concept - it knows nothing about any
particular query.

RetrievedChunk wraps a StoredChunk rather than duplicating or
re-declaring its fields: the persisted chunk's identity and content are
still whatever the persistence layer already produced, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.persistence.models import StoredChunk


@dataclass
class RetrievedChunk:
    """One chunk returned by a similarity search, in the context of one
    particular query.

    stored_chunk: the persisted chunk itself, unchanged.
    score: the similarity score (cosine similarity, in this milestone)
        between the query embedding and this chunk's embedding. Higher is
        more similar.
    rank: this result's 1-indexed position within its result list, where
        rank=1 is the most similar chunk returned for that search.
    """

    stored_chunk: StoredChunk
    score: float
    rank: int