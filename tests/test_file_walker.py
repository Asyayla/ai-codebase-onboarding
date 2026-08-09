"""
Unit tests for app/ingestion/file_walker.py.

All tests build temporary directory structures on disk (via pytest's tmp_path
fixture) rather than relying on any real, external repository.
"""

from pathlib import Path

import pytest

from app.ingestion.file_walker import walk_repository, DEFAULT_MAX_FILE_SIZE_BYTES


def _write(path: Path, content: str = "content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_includes_normal_source_files(tmp_path: Path):
    _write(tmp_path / "main.py", "print('hello')")
    _write(tmp_path / "app.js", "console.log('hi')")
    _write(tmp_path / "component.tsx", "export const X = () => null;")

    result = walk_repository(tmp_path)

    assert Path("main.py") in result
    assert Path("app.js") in result
    assert Path("component.tsx") in result


def test_includes_nested_directories(tmp_path: Path):
    _write(tmp_path / "src" / "core" / "utils.py", "x = 1")
    _write(tmp_path / "src" / "core" / "deep" / "helper.py", "y = 2")

    result = walk_repository(tmp_path)

    assert Path("src/core/utils.py") in result
    assert Path("src/core/deep/helper.py") in result


def test_excludes_denylisted_directories(tmp_path: Path):
    _write(tmp_path / "app.py", "x = 1")
    _write(tmp_path / "node_modules" / "some_pkg" / "index.js", "noise")
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main")
    _write(tmp_path / "__pycache__" / "app.cpython-311.pyc", "binary-ish")
    _write(tmp_path / ".venv" / "lib" / "site.py", "venv internals")
    _write(tmp_path / "venv" / "lib" / "site2.py", "venv internals")
    _write(tmp_path / "dist" / "bundle.js", "built output")
    _write(tmp_path / "build" / "out.py", "built output")
    _write(tmp_path / "coverage" / "report.js", "coverage output")
    _write(tmp_path / ".next" / "cache.js", "next.js cache")

    result = walk_repository(tmp_path)

    assert result == [Path("app.py")]


def test_excludes_unsupported_and_binary_files(tmp_path: Path):
    _write(tmp_path / "main.py", "x = 1")
    _write(tmp_path / "image.png", "not really binary but wrong extension")
    _write(tmp_path / "archive.zip", "zip-ish")
    _write(tmp_path / "compiled.exe", "exe-ish")
    _write(tmp_path / "notes.docx", "docx-ish")

    result = walk_repository(tmp_path)

    assert result == [Path("main.py")]


def test_includes_recognized_config_and_doc_files(tmp_path: Path):
    _write(tmp_path / "package.json", "{}")
    _write(tmp_path / "requirements.txt", "fastapi")
    _write(tmp_path / "README.md", "# Project")
    _write(tmp_path / "Dockerfile", "FROM python:3.11")

    result = walk_repository(tmp_path)

    assert set(result) == {
        Path("package.json"),
        Path("requirements.txt"),
        Path("README.md"),
        Path("Dockerfile"),
    }


def test_excludes_lockfiles_even_though_they_look_like_config(tmp_path: Path):
    _write(tmp_path / "package.json", "{}")
    _write(tmp_path / "package-lock.json", "{...huge generated content...}")
    _write(tmp_path / "yarn.lock", "# generated")

    result = walk_repository(tmp_path)

    assert result == [Path("package.json")]


def test_excludes_oversized_files(tmp_path: Path):
    small_file = tmp_path / "small.py"
    large_file = tmp_path / "large.py"
    _write(small_file, "x = 1")
    _write(large_file, "x" * 1000)

    result = walk_repository(tmp_path, max_file_size_bytes=100)

    assert Path("small.py") in result
    assert Path("large.py") not in result


def test_default_max_file_size_is_500kb():
    assert DEFAULT_MAX_FILE_SIZE_BYTES == 500_000


def test_deterministic_ordering(tmp_path: Path):
    _write(tmp_path / "b.py", "x = 1")
    _write(tmp_path / "a.py", "x = 1")
    _write(tmp_path / "src" / "z.py", "x = 1")
    _write(tmp_path / "src" / "a.py", "x = 1")

    first_run = walk_repository(tmp_path)
    second_run = walk_repository(tmp_path)

    assert first_run == second_run
    assert first_run == sorted(first_run, key=lambda p: p.as_posix())


def test_handles_unreadable_file_gracefully(tmp_path: Path):
    good_file = tmp_path / "good.py"
    unreadable_file = tmp_path / "locked.py"
    _write(good_file, "x = 1")
    _write(unreadable_file, "x = 2")

    unreadable_file.chmod(0o000)
    try:
        result = walk_repository(tmp_path)
        assert Path("good.py") in result
    finally:
        unreadable_file.chmod(0o644)


def test_symlinked_file_is_excluded(tmp_path: Path):
    real_file = tmp_path / "real.py"
    _write(real_file, "x = 1")
    symlink_file = tmp_path / "link.py"

    try:
        symlink_file.symlink_to(real_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported in this environment")

    result = walk_repository(tmp_path)

    assert Path("real.py") in result
    assert Path("link.py") not in result


def test_symlinked_directory_is_not_followed(tmp_path: Path):
    real_dir = tmp_path / "real_dir"
    _write(real_dir / "inside.py", "x = 1")
    symlink_dir = tmp_path / "link_dir"

    try:
        symlink_dir.symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported in this environment")

    result = walk_repository(tmp_path)

    assert Path("real_dir/inside.py") in result
    assert not any(str(p).startswith("link_dir") for p in result)


def test_extensible_via_extra_allowed_extensions(tmp_path: Path):
    _write(tmp_path / "main.go", "package main")

    without_extra = walk_repository(tmp_path)
    with_extra = walk_repository(tmp_path, extra_allowed_extensions={".go"})

    assert Path("main.go") not in without_extra
    assert Path("main.go") in with_extra


def test_extensible_via_extra_excluded_dirs(tmp_path: Path):
    _write(tmp_path / "keep.py", "x = 1")
    _write(tmp_path / "vendor" / "third_party.py", "x = 2")

    without_extra = walk_repository(tmp_path)
    with_extra = walk_repository(tmp_path, extra_excluded_dirs={"vendor"})

    assert Path("vendor/third_party.py") in without_extra
    assert Path("vendor/third_party.py") not in with_extra


def test_raises_on_nonexistent_root(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(NotADirectoryError):
        walk_repository(missing)