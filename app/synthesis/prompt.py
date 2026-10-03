from __future__ import annotations

from typing import List

from app.retrieval.models import RetrievedChunk

_CHUNK_SEPARATOR = "\n\n---\n\n"


def build_context(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return ""

    sections: List[str] = []
    for chunk in chunks:
        stored = chunk.stored_chunk
        header = (
            f"Source: {stored.file_path} "
            f"(chunk_id={stored.chunk_id}, score={chunk.score})"
        )
        sections.append(f"{header}\n{stored.content}")

    return _CHUNK_SEPARATOR.join(sections)