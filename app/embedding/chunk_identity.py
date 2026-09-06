"""
chunk_identity.py

Responsibility:
    Compute a deterministic, content-independent identifier for a chunk,
    given only its identity (which repository, which file, which symbol,
    where it sits) - never its source text.

    This ID is stable across re-ingestions of unchanged code, and stable
    across process restarts (unlike Python's built-in hash(), which is
    randomized per-process for str/bytes and must never be used here).

    Deliberately excluded from the ID: chunk content. A chunk's identity
    represents "this symbol, at this location" - not "this exact text".
    Editing a function's body does not change its chunk_id; only moving,
    renaming, or resizing it does. This lets a future persistence layer
    distinguish "this chunk's content changed" (update in place) from
    "this is a new chunk" (insert) using a *separate* content comparison -
    a concern this module does not implement or need to know about.
"""

from __future__ import annotations

import hashlib


def _canonicalize(*parts: str) -> str:
    """Serialize `parts` into a single string with no ambiguity between
    different input tuples.

    A naive join with a fixed delimiter (e.g. "|") is not safe here: the
    fields are free-form strings (file paths, symbol names) that could
    themselves contain any character, including whatever delimiter was
    picked - e.g. ("a|b", "c") and ("a", "b|c") would collide under plain
    "|".join(...).

    Instead, each part is length-prefixed (netstring-style: "<length>:
    <value>") before concatenation. Because each part's exact length is
    encoded immediately before it, there is exactly one way to parse the
    resulting string back into its original parts - so two different
    input tuples can never produce the same canonical string, regardless
    of what characters the parts themselves contain.
    """
    return "".join(f"{len(part)}:{part}" for part in parts)


def compute_chunk_id(
    repository_id: str,
    file_path: str,
    symbol_type: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    Compute a deterministic SHA-256 hex digest identifying a chunk by its
    identity alone - repository, file, symbol, and location. Two calls
    with identical arguments always return identical output, in this
    process or any other. Chunk content is never part of the input.

    Args:
        repository_id: identifies which repository this chunk belongs to.
        file_path: the chunk's file, relative to the repository root.
        symbol_type: e.g. "function", "class", "method", "module".
        symbol_name: e.g. "foo", "Bar.method_one", "<module>".
        start_line: 1-indexed start line, as produced by the chunker.
        end_line: 1-indexed end line, as produced by the chunker.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 digest.
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