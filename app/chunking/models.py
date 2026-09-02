"""
models.py

The data structure produced by any language chunker in app/chunking/.
Defined separately from python_chunker.py so a future javascript_chunker.py
can return the same shape without depending on the Python chunker or
duplicating this type definition - the same pattern already used in
app/parsing/models.py for parser output.

A Chunk is a retrieval-ready unit: unlike app/parsing's structural facts
(which deliberately hold no source text), a Chunk carries the actual source
text slice it represents, because that text is what the future embedding
layer will need to send to an embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    """One retrieval-ready unit of source code.

    symbol_type is one of "module", "function", "class", "method".
    symbol_name is the human-readable identity of the chunk - for a method,
    this is "ClassName.method_name", not just "method_name", so a retrieved
    chunk is self-identifying without needing to join back to its class.
    parent_symbol_name is the enclosing class name for a method chunk, and
    None for every other symbol_type.
    """

    file_path: str
    language: str
    symbol_type: str
    symbol_name: str
    start_line: int
    end_line: int
    content: str
    parent_symbol_name: Optional[str]
    docstring: Optional[str]