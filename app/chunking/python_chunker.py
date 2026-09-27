from __future__ import annotations

from typing import List, Set, Tuple

from app.chunking.models import Chunk
from app.parsing.models import ParsedFile


def _slice_by_line_range(lines: List[str], start_line: int, end_line: int) -> str:
    """1-indexed, inclusive on both ends - matches the parser's convention."""
    return "\n".join(lines[start_line - 1:end_line])


def _slice_by_line_numbers(lines: List[str], line_numbers: List[int]) -> str:
    """Build content from exact 1-indexed line numbers.

    Used for module chunks so unrelated function/class content is not included.
    """
    return "\n".join(lines[i - 1] for i in line_numbers)


def _is_meaningful_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def chunk_python_file(
    parsed_file: ParsedFile,
    source_code: str,
    language: str = "python",
) -> List[Chunk]:
    """Convert parser output and source text into retrieval-ready chunks."""
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

        # The class range already covers its methods, so adding them separately
        # would be redundant.
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