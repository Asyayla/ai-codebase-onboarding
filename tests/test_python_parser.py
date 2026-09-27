import textwrap

import pytest

from app.parsing.python_parser import parse_python_file


def _source(text: str) -> str:
    """Dedent a triple-quoted fixture before parsing."""
    return textwrap.dedent(text).lstrip("\n")


def test_single_top_level_function_no_docstring():
    source = _source(
        """
        def greet():
            pass
        """
    )

    result = parse_python_file(source, "greet.py")

    assert len(result.functions) == 1
    fn = result.functions[0]
    assert fn.name == "greet"
    assert fn.docstring is None
    assert fn.is_async is False
    assert result.parse_errors == []


def test_function_with_docstring_is_captured():
    source = _source(
        """
        def greet():
            \"\"\"Say hello.\"\"\"
            pass
        """
    )

    result = parse_python_file(source, "greet.py")

    assert result.functions[0].docstring == "Say hello."


def test_async_function_is_flagged():
    source = _source(
        """
        async def fetch_data():
            pass
        """
    )

    result = parse_python_file(source, "async_fn.py")

    assert len(result.functions) == 1
    assert result.functions[0].name == "fetch_data"
    assert result.functions[0].is_async is True


def test_class_with_two_methods():
    source = _source(
        """
        class Greeter:
            def hello(self):
                pass

            def goodbye(self):
                pass
        """
    )

    result = parse_python_file(source, "greeter.py")

    assert len(result.classes) == 1
    cls = result.classes[0]
    assert cls.name == "Greeter"
    assert len(cls.methods) == 2
    assert {m.name for m in cls.methods} == {"hello", "goodbye"}
    assert result.functions == []


def test_class_with_docstring_and_no_methods():
    source = _source(
        """
        class Empty:
            \"\"\"Nothing to see here.\"\"\"
        """
    )

    result = parse_python_file(source, "empty.py")

    assert len(result.classes) == 1
    cls = result.classes[0]
    assert cls.docstring == "Nothing to see here."
    assert cls.methods == []


def test_decorated_top_level_function_is_still_captured():
    source = _source(
        """
        @some_decorator
        def handler():
            pass
        """
    )

    result = parse_python_file(source, "handler.py")

    assert len(result.functions) == 1
    assert result.functions[0].name == "handler"


def test_decorated_method_is_still_captured_as_a_method():
    source = _source(
        """
        class Widget:
            @staticmethod
            def build():
                pass
        """
    )

    result = parse_python_file(source, "widget.py")

    assert len(result.classes) == 1
    assert len(result.classes[0].methods) == 1
    assert result.classes[0].methods[0].name == "build"
    assert result.functions == []


def test_bare_import():
    result = parse_python_file("import os\n", "mod.py")

    assert len(result.imports) == 1
    assert result.imports[0].module == "os"
    assert result.imports[0].imported_names == []


def test_dotted_bare_import():
    result = parse_python_file("import os.path\n", "mod.py")

    assert result.imports[0].module == "os.path"
    assert result.imports[0].imported_names == []


def test_comma_separated_bare_imports_become_separate_entries():
    result = parse_python_file("import os, sys\n", "mod.py")

    modules = {imp.module for imp in result.imports}
    assert modules == {"os", "sys"}
    for imp in result.imports:
        assert imp.imported_names == []


def test_from_import_with_multiple_names():
    result = parse_python_file("from typing import List, Optional\n", "mod.py")

    assert len(result.imports) == 1
    assert result.imports[0].module == "typing"
    assert result.imports[0].imported_names == ["List", "Optional"]


def test_import_alias_is_not_tracked():
    result = parse_python_file("import numpy as np\n", "mod.py")

    assert result.imports[0].module == "numpy"
    assert result.imports[0].imported_names == []


def test_from_import_alias_is_not_tracked():
    result = parse_python_file("from pathlib import Path as P\n", "mod.py")

    assert result.imports[0].module == "pathlib"
    assert result.imports[0].imported_names == ["Path"]


def test_invalid_python_syntax_returns_parse_error_not_exception():
    result = parse_python_file("def broken(:\n    pass\n", "broken.py")

    assert result.functions == []
    assert result.classes == []
    assert result.imports == []
    assert len(result.parse_errors) == 1
    assert "broken.py" in result.parse_errors[0]


def test_empty_string_returns_valid_empty_result():
    result = parse_python_file("", "empty.py")

    assert result.functions == []
    assert result.classes == []
    assert result.imports == []
    assert result.parse_errors == []


def test_comments_and_module_docstring_only_produce_no_structure():
    source = _source(
        """
        \"\"\"Module docstring.\"\"\"
        # just a comment
        """
    )

    result = parse_python_file(source, "docs_only.py")

    assert result.functions == []
    assert result.classes == []
    assert result.imports == []
    assert result.parse_errors == []


def test_non_string_input_raises_type_error():
    with pytest.raises(TypeError):
        parse_python_file(None, "whatever.py")


def test_line_numbers_are_one_indexed_and_correct():
    source = _source(
        """
        import os

        def foo():
            pass

        class Bar:
            def method(self):
                pass
        """
    )

    result = parse_python_file(source, "lines.py")

    # Tree-sitter uses zero-indexed rows; the parser exposes one-indexed lines.
    assert result.imports[0].start_line == 1
    assert result.imports[0].end_line == 1

    assert result.functions[0].start_line == 3
    assert result.functions[0].end_line == 4

    cls = result.classes[0]
    assert cls.start_line == 6
    assert cls.end_line == 8

    method = cls.methods[0]
    assert method.start_line == 7
    assert method.end_line == 8


def test_file_path_is_attached_to_result_unchanged():
    result = parse_python_file("x = 1\n", "some/nested/path.py")

    assert result.file_path == "some/nested/path.py"