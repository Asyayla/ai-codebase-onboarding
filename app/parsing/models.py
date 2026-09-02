"""
models.py

Shared, language-agnostic data structures returned by any language parser
in app/parsing/. Defined separately from python_parser.py so a future
js_ts_parser.py can return the same shapes without depending on the Python
parser or duplicating these type definitions.

These structures hold structural FACTS ONLY - names, line ranges, docstrings.
They deliberately do not hold extracted source text: turning a line range
into actual chunk content is chunking's responsibility (a later phase), not
parsing's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ImportStatement:
    """A single imported module, as it appears in one import statement.

    For `import os` and `import os.path`: module holds the dotted name,
    imported_names is empty.
    For `from typing import List, Optional`: module="typing",
    imported_names=["List", "Optional"].
    Import aliases (`as np`, `as P`) are deliberately not tracked in this
    milestone - only the real module/name is recorded.
    """

    module: str
    imported_names: List[str]
    start_line: int
    end_line: int


@dataclass
class FunctionDefinition:
    """A single function or method definition. Reused for both top-level
    functions and methods nested inside a class - a method is structurally
    a function, just nested one level deeper."""

    name: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    is_async: bool


@dataclass
class ClassDefinition:
    """A single top-level class definition, including its methods."""

    name: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    methods: List[FunctionDefinition] = field(default_factory=list)


@dataclass
class ParsedFile:
    """The full structural extraction result for one source file.

    parse_errors is populated (with imports/functions/classes left empty)
    when the source could not be parsed into a usable structure - this
    function never raises for a parsing failure, only for a genuine
    programming error (e.g. a non-string argument).
    """

    file_path: str
    imports: List[ImportStatement] = field(default_factory=list)
    functions: List[FunctionDefinition] = field(default_factory=list)
    classes: List[ClassDefinition] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)