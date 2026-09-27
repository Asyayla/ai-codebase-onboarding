from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    """A retrieval-ready unit of source code.

    symbol_type is one of "module", "function", "class", or "method".
    Method chunks use "ClassName.method_name" as symbol_name.
    parent_symbol_name is the enclosing class name for method chunks and
    None for other symbol types.
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