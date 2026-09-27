from __future__ import annotations

import hashlib


def compute_content_hash(content: str) -> str:
    """Compute a deterministic SHA-256 hash of the raw content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()