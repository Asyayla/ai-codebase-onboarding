from __future__ import annotations

from typing import List, Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from app.parsing.models import ClassDefinition, FunctionDefinition, ImportStatement, ParsedFile

_PYTHON_LANGUAGE = Language(tspython.language())


def _line_range(node: Node) -> tuple[int, int]:
    """Convert tree-sitter's 0-indexed rows to 1-indexed line numbers."""
    return node.start_point.row + 1, node.end_point.row + 1


def _node_text(node: Node, source_bytes: bytes) -> str:
    text = node.text
    if text is None:
        return ""
    return text.decode("utf-8")


def _unwrap_decorated(node: Node) -> Node:
    """Return the underlying function or class from a decorated definition."""
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner is not None:
            return inner
    return node


def _extract_docstring(
    definition_node: Node,
    source_bytes: bytes,
) -> Optional[str]:
    """Extract the string literal used as the first body statement."""
    import ast

    body = definition_node.child_by_field_name("body")
    if body is None or len(body.children) == 0:
        return None

    first_statement = body.children[0]
    if first_statement.type != "expression_statement":
        return None
    if len(first_statement.children) == 0:
        return None

    string_node = first_statement.children[0]
    if string_node.type != "string":
        return None

    text = string_node.text
    if text is None:
        return None

    try:
        value = ast.literal_eval(text.decode("utf-8"))
    except (SyntaxError, ValueError):
        return None

    return value.strip() if isinstance(value, str) else None


def _is_async(function_node: Node) -> bool:
    return any(child.type == "async" for child in function_node.children)


def _extract_function(function_node: Node, source_bytes: bytes) -> FunctionDefinition:
    # Use the function_definition range rather than the decorator range so
    # start_line refers to the actual function definition.
    name_node = function_node.child_by_field_name("name")
    start_line, end_line = _line_range(function_node)

    return FunctionDefinition(
        name=_node_text(name_node, source_bytes) if name_node else "<unknown>",
        start_line=start_line,
        end_line=end_line,
        docstring=_extract_docstring(function_node, source_bytes),
        is_async=_is_async(function_node),
    )


def _extract_class(class_node: Node, source_bytes: bytes) -> ClassDefinition:
    name_node = class_node.child_by_field_name("name")
    start_line, end_line = _line_range(class_node)

    methods: List[FunctionDefinition] = []
    body = class_node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            actual = _unwrap_decorated(child)
            if actual.type == "function_definition":
                methods.append(_extract_function(actual, source_bytes))

    return ClassDefinition(
        name=_node_text(name_node, source_bytes) if name_node else "<unknown>",
        start_line=start_line,
        end_line=end_line,
        docstring=_extract_docstring(class_node, source_bytes),
        methods=methods,
    )


def _extract_imports_from_statement(stmt_node: Node, source_bytes: bytes) -> List[ImportStatement]:
    """Extract module references from an import statement."""
    start_line, end_line = _line_range(stmt_node)
    results: List[ImportStatement] = []

    for name_node in stmt_node.children_by_field_name("name"):
        if name_node.type == "aliased_import":
            module_node = name_node.child_by_field_name("name")
        else:
            module_node = name_node

        module_text = _node_text(module_node, source_bytes) if module_node else "<unknown>"
        results.append(
            ImportStatement(
                module=module_text,
                imported_names=[],
                start_line=start_line,
                end_line=end_line,
            )
        )

    return results


def _extract_import_from_statement(
    stmt_node: Node,
    source_bytes: bytes,
) -> Optional[ImportStatement]:
    """Extract the module and imported names from a from-import statement."""
    module_node = stmt_node.child_by_field_name("module_name")
    if module_node is None:
        return None

    start_line, end_line = _line_range(stmt_node)
    imported_names: List[str] = []

    for name_node in stmt_node.children_by_field_name("name"):
        if name_node.type == "aliased_import":
            actual_name_node = name_node.child_by_field_name("name")
        else:
            actual_name_node = name_node

        if actual_name_node is not None:
            imported_names.append(_node_text(actual_name_node, source_bytes))

    return ImportStatement(
        module=_node_text(module_node, source_bytes),
        imported_names=imported_names,
        start_line=start_line,
        end_line=end_line,
    )


def parse_python_file(source_code: str, file_path: str) -> ParsedFile:
    """Extract the top-level structure of a Python source file."""
    if not isinstance(source_code, str):
        raise TypeError(f"source_code must be a str, got {type(source_code).__name__}")

    source_bytes = source_code.encode("utf-8")
    parser = Parser(_PYTHON_LANGUAGE)
    tree = parser.parse(source_bytes)
    root = tree.root_node

    if root.has_error:
        return ParsedFile(
            file_path=file_path,
            parse_errors=[f"Failed to parse '{file_path}': source contains invalid Python syntax."],
        )

    imports: List[ImportStatement] = []
    functions: List[FunctionDefinition] = []
    classes: List[ClassDefinition] = []

    for top_level_node in root.children:
        node = _unwrap_decorated(top_level_node)

        if node.type == "import_statement":
            imports.extend(_extract_imports_from_statement(node, source_bytes))
        elif node.type == "import_from_statement":
            import_stmt = _extract_import_from_statement(node, source_bytes)
            if import_stmt is not None:
                imports.append(import_stmt)
        elif node.type == "function_definition":
            functions.append(_extract_function(node, source_bytes))
        elif node.type == "class_definition":
            classes.append(_extract_class(node, source_bytes))

    return ParsedFile(
        file_path=file_path,
        imports=imports,
        functions=functions,
        classes=classes,
        parse_errors=[],
    )