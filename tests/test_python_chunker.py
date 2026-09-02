"""
Unit tests for app/chunking/python_chunker.py.

All fixtures are small Python source strings, parsed with the real
parse_python_file (never a hand-built ParsedFile), then passed into
chunk_python_file - exercising the actual parser/chunker boundary rather
than a mocked one. No disk files, no network, matching the existing test
style in this project.
"""

import textwrap

from app.chunking.python_chunker import chunk_python_file
from app.parsing.python_parser import parse_python_file


def _source(text: str) -> str:
    """Dedent a triple-quoted fixture so indentation in the test file
    itself doesn't leak into the Python source being parsed."""
    return textwrap.dedent(text).lstrip("\n")


def _chunk(source: str, file_path: str = "sample.py"):
    parsed = parse_python_file(source, file_path)
    return chunk_python_file(parsed, source)


def test_single_top_level_function_produces_one_function_chunk():
    source = _source(
        """
        def greet():
            print("hi")
        """
    )

    chunks = _chunk(source)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.symbol_type == "function"
    assert chunk.symbol_name == "greet"
    assert chunk.parent_symbol_name is None
    assert chunk.language == "python"
    assert chunk.file_path == "sample.py"
    assert chunk.start_line == 1
    assert chunk.end_line == 2
    assert chunk.content == 'def greet():\n    print("hi")'


def test_class_with_multiple_methods_produces_class_and_method_chunks():
    source = _source(
        """
        class Greeter:
            def hello(self):
                pass

            def goodbye(self):
                pass
        """
    )

    chunks = _chunk(source)

    class_chunks = [c for c in chunks if c.symbol_type == "class"]
    method_chunks = [c for c in chunks if c.symbol_type == "method"]

    assert len(class_chunks) == 1
    assert class_chunks[0].symbol_name == "Greeter"
    assert class_chunks[0].parent_symbol_name is None

    assert len(method_chunks) == 2
    method_names = {c.symbol_name for c in method_chunks}
    assert method_names == {"Greeter.hello", "Greeter.goodbye"}
    for method_chunk in method_chunks:
        assert method_chunk.parent_symbol_name == "Greeter"

    # No separate top-level function chunks - methods are not functions.
    assert [c for c in chunks if c.symbol_type == "function"] == []


def test_class_with_zero_methods_produces_only_a_class_chunk():
    source = _source(
        """
        class Empty:
            \"\"\"Nothing here.\"\"\"
        """
    )

    chunks = _chunk(source)

    assert len(chunks) == 1
    assert chunks[0].symbol_type == "class"
    assert chunks[0].symbol_name == "Empty"
    assert chunks[0].docstring == "Nothing here."


def test_function_with_trailing_module_level_code_produces_two_chunks():
    source = _source(
        """
        def main():
            pass

        if __name__ == "__main__":
            main()
        """
    )

    chunks = _chunk(source)

    function_chunks = [c for c in chunks if c.symbol_type == "function"]
    module_chunks = [c for c in chunks if c.symbol_type == "module"]

    assert len(function_chunks) == 1
    assert function_chunks[0].symbol_name == "main"
    assert function_chunks[0].start_line == 1
    assert function_chunks[0].end_line == 2

    assert len(module_chunks) == 1
    module_chunk = module_chunks[0]
    assert module_chunk.symbol_name == "<module>"
    # The complement region starts at line 3 (the blank line separating the
    # function from the trailing block) and ends at line 5. The blank line
    # is not "meaningful" on its own, but it IS part of the uncovered
    # region, so it is preserved in the module chunk's content rather than
    # stripped out - only its presence/absence doesn't affect whether a
    # module chunk gets emitted at all.
    assert module_chunk.start_line == 3
    assert module_chunk.end_line == 5
    assert module_chunk.content == '\nif __name__ == "__main__":\n    main()'
    # The function's own lines must not leak into the module chunk.
    assert "def main" not in module_chunk.content


def test_imports_only_file_produces_one_module_chunk():
    source = "import os\nimport sys\n"

    chunks = _chunk(source)

    assert len(chunks) == 1
    assert chunks[0].symbol_type == "module"
    assert chunks[0].symbol_name == "<module>"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2
    assert chunks[0].content == "import os\nimport sys"


def test_empty_file_produces_zero_chunks():
    chunks = _chunk("")

    assert chunks == []


def test_file_with_only_comments_and_blank_lines_produces_zero_chunks():
    source = _source(
        """
        # just a comment

        # another comment
        """
    )

    chunks = _chunk(source)

    assert chunks == []


def test_module_docstring_only_file_produces_one_module_chunk():
    source = _source(
        """
        \"\"\"Module docstring.\"\"\"
        """
    )

    chunks = _chunk(source)

    assert len(chunks) == 1
    assert chunks[0].symbol_type == "module"


def test_parse_errors_produce_zero_chunks_with_no_fallback():
    source = "def broken(:\n    pass\n"
    parsed = parse_python_file(source, "broken.py")
    assert parsed.parse_errors != []  # sanity check on the parser's own behavior

    chunks = chunk_python_file(parsed, source)

    assert chunks == []


def test_content_is_sliced_exactly_from_source_lines():
    source = _source(
        """
        import os

        def foo():
            x = 1
            return x

        class Bar:
            def method(self):
                return 1
        """
    )

    chunks = _chunk(source)
    lines = source.splitlines()

    for chunk in chunks:
        if chunk.symbol_type == "module":
            # The module chunk's covered region can be non-contiguous (here:
            # a blank line after the import, and a blank line between the
            # function and the class, with the function/class lines
            # excluded in between). Its content is reconstructed from only
            # the actual uncovered line numbers, not a single contiguous
            # start:end slice - a plain lines[start-1:end] slice would
            # incorrectly include the function/class source text that
            # falls between the first and last uncovered line.
            continue
        expected = "\n".join(lines[chunk.start_line - 1:chunk.end_line])
        assert chunk.content == expected


def test_module_chunk_content_excludes_lines_covered_by_functions_or_classes():
    source = _source(
        """
        import os

        def foo():
            x = 1
            return x

        class Bar:
            def method(self):
                return 1
        """
    )

    chunks = _chunk(source)
    module_chunk = next(c for c in chunks if c.symbol_type == "module")

    # "import os" and the surrounding blank lines are module-level; the
    # function and class bodies must not appear inside this chunk's
    # content, even though they fall between the module chunk's own
    # start_line and end_line.
    assert "import os" in module_chunk.content
    assert "def foo" not in module_chunk.content
    assert "class Bar" not in module_chunk.content


def test_nested_function_is_not_a_separate_chunk():
    source = _source(
        """
        def outer():
            def inner():
                return 1
            return inner()
        """
    )

    chunks = _chunk(source)

    assert len(chunks) == 1
    assert chunks[0].symbol_type == "function"
    assert chunks[0].symbol_name == "outer"
    # The nested function's source text is part of the outer chunk's
    # content, since the current parser does not extract nested functions
    # as their own symbols.
    assert "def inner" in chunks[0].content
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 4


def test_class_chunk_spans_full_body_including_methods():
    source = _source(
        """
        class Widget:
            def build(self):
                return "built"
        """
    )

    chunks = _chunk(source)
    class_chunk = next(c for c in chunks if c.symbol_type == "class")
    method_chunk = next(c for c in chunks if c.symbol_type == "method")

    assert class_chunk.start_line == 1
    assert class_chunk.end_line == 3
    assert "def build" in class_chunk.content
    assert method_chunk.symbol_name == "Widget.build"
    assert method_chunk.start_line == 2
    assert method_chunk.end_line == 3


def test_docstring_is_carried_onto_the_chunk():
    source = _source(
        """
        def greet():
            \"\"\"Say hello.\"\"\"
            pass
        """
    )

    chunks = _chunk(source)

    assert chunks[0].docstring == "Say hello."


def test_module_chunk_has_no_docstring_field_populated():
    source = "import os\n"

    chunks = _chunk(source)

    assert chunks[0].symbol_type == "module"
    assert chunks[0].docstring is None


def test_language_defaults_to_python():
    source = "def foo():\n    pass\n"

    chunks = _chunk(source)

    assert chunks[0].language == "python"


def test_language_can_be_overridden():
    source = "def foo():\n    pass\n"
    parsed = parse_python_file(source, "sample.py")

    chunks = chunk_python_file(parsed, source, language="python3")

    assert chunks[0].language == "python3"