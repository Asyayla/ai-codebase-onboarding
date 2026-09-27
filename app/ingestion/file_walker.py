from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union


LANGUAGE_EXTENSIONS: Dict[str, Set[str]] = {
    "python": {".py", ".pyi"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx"},
}

DOCUMENTATION_EXTENSIONS: Set[str] = {".md", ".rst", ".txt"}

CONFIG_AND_DOC_FILENAMES: Set[str] = {
    "package.json",
    "tsconfig.json",
    "jsconfig.json",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Makefile",
    ".env.example",
    "README",
    "README.md",
    "LICENSE",
}

EXCLUDED_FILENAMES: Set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}

EXCLUDED_DIR_NAMES: Set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    "coverage",
    ".next",
}

DEFAULT_MAX_FILE_SIZE_BYTES: int = 500_000


def _build_allowed_extensions(
    extra_allowed_extensions: Optional[Set[str]] = None,
) -> Set[str]:
    """Build the allow-list of recognized file extensions."""
    allowed: Set[str] = set()
    for ext_set in LANGUAGE_EXTENSIONS.values():
        allowed |= ext_set
    allowed |= DOCUMENTATION_EXTENSIONS

    if extra_allowed_extensions:
        allowed |= extra_allowed_extensions

    return allowed


def _is_relevant_file(
    path: Path,
    allowed_extensions: Set[str],
) -> bool:
    name = path.name

    if name in EXCLUDED_FILENAMES:
        return False

    if name in CONFIG_AND_DOC_FILENAMES:
        return True

    return path.suffix in allowed_extensions


def walk_repository(
    root: Union[str, Path],
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    extra_excluded_dirs: Optional[Set[str]] = None,
    extra_allowed_extensions: Optional[Set[str]] = None,
) -> List[Path]:
    """Return relevant repository files as sorted paths relative to root."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"root is not a directory: {root_path}")

    excluded_dirs = EXCLUDED_DIR_NAMES | (extra_excluded_dirs or set())
    allowed_extensions = _build_allowed_extensions(extra_allowed_extensions)

    results: List[Path] = []

    for current_dir, dir_names, file_names in os.walk(root_path, followlinks=False):
        current_dir_path = Path(current_dir)

        dir_names[:] = sorted(
            d
            for d in dir_names
            if d not in excluded_dirs and not (current_dir_path / d).is_symlink()
        )

        for file_name in sorted(file_names):
            file_path = current_dir_path / file_name

            if file_path.is_symlink():
                continue

            if not _is_relevant_file(file_path, allowed_extensions):
                continue

            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size > max_file_size_bytes:
                continue

            results.append(file_path.relative_to(root_path))

    results.sort(key=lambda p: p.as_posix())
    return results