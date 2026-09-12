"""
content_hash.py

Responsibility:
    Compute a deterministic hash of a chunk's content, for the persistence
    layer's own use in re-ingestion change detection (has this chunk's
    text changed since it was last stored?).

    This is the deliberate counterpart to app/embedding/chunk_identity.py:
    chunk_id there is content-INDEPENDENT (identity survives content
    edits); content_hash here is content-DEPENDENT (any change in content
    changes the hash). The two serve different, complementary purposes and
    neither should be confused for the other.
"""

from __future__ import annotations

import hashlib


def compute_content_hash(content: str) -> str:
    """
    Compute a deterministic SHA-256 hex digest of `content`.

    The content is hashed exactly as given - no stripping, casing changes,
    whitespace normalization, or any other transformation is applied
    before hashing. Two chunks are considered to have "the same content"
    only if their raw text is byte-for-byte identical.

    Args:
        content: the chunk's source text. May be an empty string.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()