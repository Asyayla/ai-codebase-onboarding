"""
embedding_service.py

Responsibility:
    Transform Chunk objects into EmbeddedChunk objects (or, for chunks that
    could not be embedded, EmbeddingFailure objects), by batching their
    content through an EmbeddingProvider.

    This module owns: chunk_id computation (delegated to the existing
    compute_chunk_id - never reimplemented here), batching, extracting
    chunk.content as provider input, calling the provider, validating its
    response, and assembling EmbeddedChunk/EmbeddingFailure/EmbeddingResult.

    This module does NOT know about persistence, PostgreSQL, pgvector,
    repository storage, deduplication, or re-ingestion comparisons - those
    are the responsibility of a future persistence/orchestration layer.
    embed_chunks() is a stateless, pure transformation: calling it twice
    with identical input makes identical provider calls both times: it
    does not check whether a chunk was "already embedded" anywhere.
"""

from __future__ import annotations

from typing import List

from app.chunking.models import Chunk
from app.embedding.chunk_identity import compute_chunk_id
from app.embedding.models import EmbeddedChunk, EmbeddingFailure, EmbeddingResult
from app.embedding.provider import EmbeddingProvider


def _make_batches(chunks: List[Chunk], batch_size: int) -> List[List[Chunk]]:
    """Split chunks into ordered, non-overlapping batches of at most
    batch_size, preserving input order. Never produces an empty batch."""
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
    """
    Embed a list of chunks using `provider`, in batches of `batch_size`.

    This is a stateless transformation: it does not check, or care,
    whether any of these chunks have been embedded before. Deduplication
    and re-ingestion comparisons belong to a future persistence layer, not
    here.

    Note: batch_size=16 is an arbitrary placeholder default, not a
    researched limit for any specific real embedding provider. Callers
    embedding against a real provider should pass an explicit batch_size
    once that provider's actual limits are known.

    Args:
        chunks: the chunks to embed, in the order they should be
            processed and reported back in.
        provider: anything satisfying the EmbeddingProvider Protocol.
            Only provider.embed_batch, provider.model_name, and
            provider.dimensions are used - this function never inspects
            or assumes anything else about the provider.
        repository_id: identifies which repository these chunks belong
            to; stamped onto every resulting EmbeddedChunk/EmbeddingFailure
            and used as an input to chunk_id computation. Not validated
            or interpreted in any way here.
        batch_size: maximum number of chunks sent to the provider in a
            single embed_batch call. Must be a positive integer.

    Returns:
        An EmbeddingResult whose `embedded` and `failures` lists both
        preserve the original relative order of `chunks`. A chunk always
        appears in exactly one of the two lists, never both, never
        neither.

    Raises:
        ValueError: if batch_size is not a positive integer. Raised
            immediately, before any provider calls are made.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

    if not chunks:
        return EmbeddingResult(embedded=[], failures=[])

    # Chunk IDs are content-independent (see chunk_identity.py) and cheap
    # to compute, so they're derived up front for every chunk, before any
    # provider interaction. This makes the ID available on both the
    # success and failure paths below without recomputing it twice.
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

        # Only chunk.content ever leaves this function and reaches the
        # provider - no other Chunk field is used to build provider input.
        texts = [chunk.content for chunk in batch]

        # This is the single call site for provider.embed_batch, so a
        # future retry/backoff wrapper can be introduced around exactly
        # this line without restructuring the surrounding batching logic.
        try:
            vectors = provider.embed_batch(texts)
        except Exception as exc:  # noqa: BLE001 - provider failures are data, not bugs in this layer
            for chunk, chunk_id in zip(batch, batch_chunk_ids):
                failures.append(_failure_for(chunk, chunk_id, str(exc)))
            continue

        if len(vectors) != len(texts):
            # No safe way to determine which vector belongs to which
            # chunk when the count itself is wrong - the whole batch is
            # untrustworthy, not just some of it.
            error_message = (
                f"provider returned {len(vectors)} vectors for "
                f"{len(texts)} input texts"
            )
            for chunk, chunk_id in zip(batch, batch_chunk_ids):
                failures.append(_failure_for(chunk, chunk_id, error_message))
            continue

        # Count is correct, so positional correlation is trustworthy here:
        # each chunk is evaluated independently from this point on, and a
        # bad vector for one chunk does not affect its batch-mates.
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