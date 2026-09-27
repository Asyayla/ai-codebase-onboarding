import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from app.ingestion.clone import RepositoryCloneError
from app.ingestion.repository_ingestion import ingest_repository


def _run_git(args, cwd):
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert result.returncode == 0, f"git {args} failed:\n{result.stderr}"
    return result


@pytest.fixture
def local_git_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "fixture_repo"
    repo_dir.mkdir()

    _run_git(["init"], cwd=repo_dir)

    (repo_dir / "main.py").write_text("print('hello world')")
    (repo_dir / "README.md").write_text("# Fixture Project")
    (repo_dir / "image.png").write_text("not a real png, just wrong extension")

    node_modules_dir = repo_dir / "node_modules" / "some_pkg"
    node_modules_dir.mkdir(parents=True)
    (node_modules_dir / "index.js").write_text("module.exports = {};")

    _run_git(["add", "-A"], cwd=repo_dir)
    _run_git(
        [
            "-c", "user.email=test@example.com",
            "-c", "user.name=Test User",
            "commit", "-m", "initial commit",
        ],
        cwd=repo_dir,
    )

    return repo_dir


def test_ingest_repository_returns_files_discovered_by_walker(local_git_repo: Path):
    result = ingest_repository(str(local_git_repo))

    assert Path("main.py") in result
    assert Path("README.md") in result


def test_ingest_repository_returns_relative_paths(local_git_repo: Path):
    result = ingest_repository(str(local_git_repo))

    for file_path in result:
        assert not file_path.is_absolute()


def test_ingest_repository_excludes_unsupported_files(local_git_repo: Path):
    result = ingest_repository(str(local_git_repo))

    assert Path("image.png") not in result


def test_ingest_repository_excludes_denylisted_directories(local_git_repo: Path):
    result = ingest_repository(str(local_git_repo))

    assert not any(str(p).startswith("node_modules") for p in result)


def test_ingest_repository_invalid_source_raises_clone_error(tmp_path: Path):
    nonexistent_source = str(tmp_path / "does_not_exist")

    with pytest.raises(RepositoryCloneError):
        ingest_repository(nonexistent_source)


def test_ingest_repository_calls_clone_then_walk_in_order():
    # Verify composition without re-testing clone_repository or walk_repository.
    fake_cloned_path = Path("/fake/cloned/path")
    fake_walk_result = [Path("main.py"), Path("README.md")]

    with patch(
        "app.ingestion.repository_ingestion.clone_repository",
        return_value=fake_cloned_path,
    ) as mock_clone, patch(
        "app.ingestion.repository_ingestion.walk_repository",
        return_value=fake_walk_result,
    ) as mock_walk:
        result = ingest_repository("https://github.com/example/repo.git")

    mock_clone.assert_called_once_with("https://github.com/example/repo.git")
    mock_walk.assert_called_once_with(fake_cloned_path)
    assert result == fake_walk_result