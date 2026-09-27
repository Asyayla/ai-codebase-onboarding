import subprocess
from pathlib import Path

import pytest

import app.ingestion.clone as clone_module
from app.ingestion.clone import RepositoryCloneError, clone_repository


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

    (repo_dir / "first.txt").write_text("first commit content")
    _run_git(["add", "first.txt"], cwd=repo_dir)
    _run_git(
        [
            "-c", "user.email=test@example.com",
            "-c", "user.name=Test User",
            "commit", "-m", "first commit",
        ],
        cwd=repo_dir,
    )

    (repo_dir / "second.txt").write_text("second commit content")
    _run_git(["add", "second.txt"], cwd=repo_dir)
    _run_git(
        [
            "-c", "user.email=test@example.com",
            "-c", "user.name=Test User",
            "commit", "-m", "second commit",
        ],
        cwd=repo_dir,
    )

    return repo_dir


def test_clone_returns_existing_directory(local_git_repo: Path):
    destination = clone_repository(str(local_git_repo))

    assert destination.exists()
    assert destination.is_dir()


def test_cloned_repository_contains_expected_files(local_git_repo: Path):
    destination = clone_repository(str(local_git_repo))

    assert (destination / "first.txt").exists()
    assert (destination / "second.txt").exists()
    assert (destination / "first.txt").read_text() == "first commit content"
    assert (destination / "second.txt").read_text() == "second commit content"


def test_clone_is_shallow_with_exactly_one_commit(local_git_repo: Path):
    destination = clone_repository(str(local_git_repo))

    result = subprocess.run(
        ["git", "-C", str(destination), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
        shell=False,
    )

    assert result.returncode == 0
    # The source has two commits; a depth-1 clone should have one.
    assert result.stdout.strip() == "1"


def test_clone_nonexistent_source_raises_clear_error(tmp_path: Path):
    nonexistent_source = str(tmp_path / "does_not_exist")

    with pytest.raises(RepositoryCloneError) as exc_info:
        clone_repository(nonexistent_source)

    message = str(exc_info.value)
    assert "git clone failed" in message
    assert nonexistent_source in message


def test_two_clones_produce_different_destinations(local_git_repo: Path):
    destination_one = clone_repository(str(local_git_repo))
    destination_two = clone_repository(str(local_git_repo))

    assert destination_one != destination_two
    assert destination_one.exists()
    assert destination_two.exists()


def test_failed_clone_removes_partial_temp_directory(
    tmp_path: Path,
    monkeypatch,
):
    created_dirs = []
    original_mkdtemp = clone_module.tempfile.mkdtemp

    def spy_mkdtemp(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        created_dirs.append(path)
        return path

    monkeypatch.setattr(clone_module.tempfile, "mkdtemp", spy_mkdtemp)

    nonexistent_source = str(tmp_path / "does_not_exist")
    with pytest.raises(RepositoryCloneError):
        clone_repository(nonexistent_source)

    assert len(created_dirs) == 1
    assert not Path(created_dirs[0]).exists()