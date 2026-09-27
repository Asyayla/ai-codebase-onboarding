from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ImportStatement:
    """A single imported module and its imported names.

    Import aliases are not tracked.
    """

    module: str
    imported_names: List[str]
    start_line: int
    end_line: int


@dataclass
class FunctionDefinition:
    """A function or method definition."""

    name: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    is_async: bool


@dataclass
class ClassDefinition:
    """A top-level class definition, including its methods."""

    name: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    methods: List[FunctionDefinition] = field(default_factory=list)


@dataclass
class ParsedFile:
    """The structural extraction result for one source file.

    Parse failures are recorded in parse_errors instead of raising.
    """

    file_path: str
    imports: List[ImportStatement] = field(default_factory=list)
    functions: List[FunctionDefinition] = field(default_factory=list)
    classes: List[ClassDefinition] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)