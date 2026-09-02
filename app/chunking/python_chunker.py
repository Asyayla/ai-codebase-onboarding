"""
python_chunker.py

Responsibility:
    Convert an already-produced ParsedFile (app/parsing/models.py) plus the
    original source text into a list of retrieval-ready Chunk objects.

    This module does NOT parse source code itself - it consumes exactly
    what app/parsing/python_parser.py already extracted (names, line
    ranges, docstrings) and slices the corresponding text out of
    source_code. It does NOT read files from disk, does NOT talk to an
    embedding model or a database, and does NOT dispatch between languages.

Contract:
    chunk_python_file(parsed_file, source_code, language="python") -> List[Chunk]

    - If parsed_file.parse_errors is non-empty, returns [] immediately.
      No naive line-based fallback chunking is ever performed.
    - One Chunk per top-level function (symbol_type="function").
    - One Chunk per top-level class, spanning the whole class body
      (symbol_type="class"), PLUS one Chunk per method inside it
      (symbol_type="method", symbol_name="ClassName.method_name",
      parent_symbol_name="ClassName").
    - At most one additional Chunk (symbol_type="module") covering
      meaningful lines not covered by any top-level function/class range.
      Emitted only if such lines exist and at least one is non-blank,
      non-comment-only text.
    - All line numbers are 1-indexed, inherited unchanged from the parser.
    - Chunk.content is always built directly from source_code's lines
      using those line numbers - never re-derived or guessed.
"""

from __future__ import annotations

from typing import List, Set, Tuple

from app.chunking.models import Chunk
from app.parsing.models import ParsedFile


def _slice_by_line_range(lines: List[str], start_line: int, end_line: int) -> str:
    """1-indexed, inclusive on both ends - matches the parser's convention."""
    return "\n".join(lines[start_line - 1:end_line])


def _slice_by_line_numbers(lines: List[str], line_numbers: List[int]) -> str:
    """Like _slice_by_line_range, but for a possibly non-contiguous set of
    1-indexed line numbers. Used only for the module-level chunk, whose
    complement region (everything not covered by a function/class) may be
    split across more than one region of the file - e.g. imports before a
    function and a trailing `if __name__ == "__main__":` block after it.
    Reconstructing content from exactly these line numbers (rather than a
    single min-to-max slice) keeps the module chunk free of unrelated
    function/class source text that would otherwise fall inside that
    min-to-max range."""
    return "\n".join(lines[i - 1] for i in line_numbers)


def _is_meaningful_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def chunk_python_file(
    parsed_file: ParsedFile,
    source_code: str,
    language: str = "python",
) -> List[Chunk]:
    """
    Convert parser output into retrieval-ready chunks.

    Args:
        parsed_file: the structural extraction result from
            parse_python_file. Never re-parsed here.
        source_code: the exact source text parsed_file was extracted from.
            Used only to slice out chunk content by line number.
        language: attached to every produced Chunk.

    Returns:
        A list of Chunk objects. Empty if parsed_file.parse_errors is
        non-empty, or if the file has no functions/classes and no
        meaningful module-level content.
    """
    if parsed_file.parse_errors:
        return []

    lines = source_code.splitlines()
    chunks: List[Chunk] = []
    covered_ranges: List[Tuple[int, int]] = []

    for function in parsed_file.functions:
        chunks.append(
            Chunk(
                file_path=parsed_file.file_path,
                language=language,
                symbol_type="function",
                symbol_name=function.name,
                start_line=function.start_line,
                end_line=function.end_line,
                content=_slice_by_line_range(lines, function.start_line, function.end_line),
                parent_symbol_name=None,
                docstring=function.docstring,
            )
        )
        covered_ranges.append((function.start_line, function.end_line))

    for cls in parsed_file.classes:
        chunks.append(
            Chunk(
                file_path=parsed_file.file_path,
                language=language,
                symbol_type="class",
                symbol_name=cls.name,
                start_line=cls.start_line,
                end_line=cls.end_line,
                content=_slice_by_line_range(lines, cls.start_line, cls.end_line),
                parent_symbol_name=None,
                docstring=cls.docstring,
            )
        )
        # The class's own range already spans all of its methods, so methods
        # are NOT added to covered_ranges separately - doing so would be
        # redundant and could only ever shrink the (already correct)
        # covered region.
        covered_ranges.append((cls.start_line, cls.end_line))

        for method in cls.methods:
            chunks.append(
                Chunk(
                    file_path=parsed_file.file_path,
                    language=language,
                    symbol_type="method",
                    symbol_name=f"{cls.name}.{method.name}",
                    start_line=method.start_line,
                    end_line=method.end_line,
                    content=_slice_by_line_range(lines, method.start_line, method.end_line),
                    parent_symbol_name=cls.name,
                    docstring=method.docstring,
                )
            )

    covered_line_numbers: Set[int] = set()
    for start, end in covered_ranges:
        covered_line_numbers.update(range(start, end + 1))

    module_line_numbers = [
        line_number
        for line_number in range(1, len(lines) + 1)
        if line_number not in covered_line_numbers
    ]

    has_meaningful_module_content = any(
        _is_meaningful_line(lines[line_number - 1]) for line_number in module_line_numbers
    )

    if module_line_numbers and has_meaningful_module_content:
        chunks.append(
            Chunk(
                file_path=parsed_file.file_path,
                language=language,
                symbol_type="module",
                symbol_name="<module>",
                start_line=module_line_numbers[0],
                end_line=module_line_numbers[-1],
                content=_slice_by_line_numbers(lines, module_line_numbers),
                parent_symbol_name=None,
                docstring=None,
            )
        )

    return chunks