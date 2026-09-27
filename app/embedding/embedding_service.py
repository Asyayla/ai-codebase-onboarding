from __future__ import annotations

from typing import List

from app.chunking.models import Chunk
from app.embedding.chunk_identity import compute_chunk_id
from app.embedding.models import EmbeddedChunk, EmbeddingFailure, EmbeddingResult
from app.embedding.provider import EmbeddingProvider


def _make_batches(chunks: List[Chunk], batch_size: int) -> List[List[Chunk]]:
    """Split chunks into ordered batches of at most batch_size."""
    return [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]


def _failure_for(chunk: Chunk, chunk_id: str, error_message: str) -> EmbeddingFailure:
    return EmbeddingFailure(
        chunk_id=chunk_id,
        file_path=chunk.file_path,
        symbol_name=chunk.symbol_name,
        error_message=error_message,
    )


def embed_chunks(
    chunks: List[Chunk],
    provider: EmbeddingProvider,
    repository_id: str,
    batch_size: int = 16,
) -> EmbeddingResult:
    """Embed chunks in batches and return successful and failed results."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

    if not chunks:
        return EmbeddingResult(embedded=[], failures=[])

    # Compute IDs before provider calls so both success and failure paths
    # can reuse the same IDs.
    chunk_ids = [
        compute_chunk_id(
            repository_id,
            chunk.file_path,
            chunk.symbol_type,
            chunk.symbol_name,
            chunk.start_line,
            chunk.end_line,
        )
        for chunk in chunks
    ]

    embedded: List[EmbeddedChunk] = []
    failures: List[EmbeddingFailure] = []

    batch_start_index = 0
    for batch in _make_batches(chunks, batch_size):
        batch_chunk_ids = chunk_ids[batch_start_index:batch_start_index + len(batch)]
        batch_start_index += len(batch)

        texts = [chunk.content for chunk in batch]

        try:
            vectors = provider.embed_batch(texts)
        except Exception as exc:  # noqa: BLE001 - provider failures are data
            for chunk, chunk_id in zip(batch, batch_chunk_ids):
                failures.append(_failure_for(chunk, chunk_id, str(exc)))
            continue

        if len(vectors) != len(texts):
            # A count mismatch makes the entire batch unreliable because
            # vector-to-chunk positional mapping is no longer trustworthy.
            error_message = (
                f"provider returned {len(vectors)} vectors for "
                f"{len(texts)} input texts"
            )
            for chunk, chunk_id in zip(batch, batch_chunk_ids):
                failures.append(_failure_for(chunk, chunk_id, error_message))
            continue

        for chunk, chunk_id, vector in zip(batch, batch_chunk_ids, vectors):
            if len(vector) != provider.dimensions:
                error_message = (
                    f"provider returned a vector with {len(vector)} dimensions, "
                    f"expected {provider.dimensions}"
                )
                failures.append(_failure_for(chunk, chunk_id, error_message))
                continue

            embedded.append(
                EmbeddedChunk(
                    chunk_id=chunk_id,
                    repository_id=repository_id,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    symbol_type=chunk.symbol_type,
                    symbol_name=chunk.symbol_name,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    parent_symbol_name=chunk.parent_symbol_name,
                    docstring=chunk.docstring,
                    embedding=vector,
                    embedding_model=provider.model_name,
                )
            )

    return EmbeddingResult(embedded=embedded, failures=failures)