from __future__ import annotations

import hashlib


def _canonicalize(*parts: str) -> str:
    """Serialize parts with length prefixes to avoid delimiter ambiguity."""
    return "".join(f"{len(part)}:{part}" for part in parts)


def compute_chunk_id(
    repository_id: str,
    file_path: str,
    symbol_type: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
) -> str:
    """Compute a deterministic SHA-256 ID from chunk identity.

    Chunk content is intentionally excluded so body changes do not change
    the chunk ID.
    """
    canonical = _canonicalize(
        repository_id,
        file_path,
        symbol_type,
        symbol_name,
        str(start_line),
        str(end_line),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()