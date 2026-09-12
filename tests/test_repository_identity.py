"""
Unit tests for app/ingestion/repository_identity.py.

All tests are pure/offline - no network, no git calls. Local-path tests
use pytest's tmp_path fixture and monkeypatch.chdir, matching the existing
test style in this project.
"""

from pathlib import Path

from app.ingestion.repository_identity import compute_repository_id


def test_normal_remote_url_is_returned_as_is():
    result = compute_repository_id("https://github.com/octocat/Hello-World")

    assert result == "https://github.com/octocat/Hello-World"


def test_trailing_slash_is_removed():
    with_slash = compute_repository_id("https://github.com/octocat/Hello-World/")
    without_slash = compute_repository_id("https://github.com/octocat/Hello-World")

    assert with_slash == without_slash


def test_trailing_git_suffix_is_removed():
    with_git_suffix = compute_repository_id("https://github.com/octocat/Hello-World.git")
    without_git_suffix = compute_repository_id("https://github.com/octocat/Hello-World")

    assert with_git_suffix == without_git_suffix


def test_scheme_casing_is_normalized():
    result = compute_repository_id("HTTPS://github.com/octocat/Hello-World")

    assert result.startswith("https://")


def test_hostname_casing_is_normalized():
    result = compute_repository_id("https://GitHub.COM/octocat/Hello-World")

    assert result == "https://github.com/octocat/Hello-World"


def test_repository_path_casing_is_preserved():
    """Only scheme and host are case-normalized - the repository path
    itself is case-sensitive on real hosting providers and must not be
    altered."""
    result = compute_repository_id("https://github.com/OctoCat/Hello-World")

    assert result == "https://github.com/OctoCat/Hello-World"


def test_equivalent_remote_url_forms_produce_the_same_id():
    forms = [
        "https://github.com/octocat/Hello-World",
        "https://github.com/octocat/Hello-World/",
        "https://github.com/octocat/Hello-World.git",
        "HTTPS://GitHub.COM/octocat/Hello-World.git",
        "https://GITHUB.com/octocat/Hello-World/",
    ]

    resolved_ids = {compute_repository_id(form) for form in forms}

    assert len(resolved_ids) == 1


def test_relative_and_absolute_local_paths_resolve_to_the_same_id(tmp_path: Path, monkeypatch):
    repo_dir = tmp_path / "my_repo"
    repo_dir.mkdir()

    monkeypatch.chdir(tmp_path)

    from_relative = compute_repository_id("my_repo")
    from_absolute = compute_repository_id(str(repo_dir))

    assert from_relative == from_absolute


def test_local_path_resolves_to_an_absolute_path(tmp_path: Path, monkeypatch):
    repo_dir = tmp_path / "my_repo"
    repo_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    result = compute_repository_id("my_repo")

    assert Path(result).is_absolute()
    assert result == str(repo_dir.resolve())


def test_distinct_remote_repositories_remain_distinct():
    first = compute_repository_id("https://github.com/octocat/Hello-World")
    second = compute_repository_id("https://github.com/octocat/Other-Repo")

    assert first != second


def test_distinct_local_repositories_remain_distinct(tmp_path: Path):
    first_dir = tmp_path / "repo_one"
    second_dir = tmp_path / "repo_two"
    first_dir.mkdir()
    second_dir.mkdir()

    first_id = compute_repository_id(str(first_dir))
    second_id = compute_repository_id(str(second_dir))

    assert first_id != second_id


def test_remote_and_local_forms_do_not_collide(tmp_path: Path):
    remote_id = compute_repository_id("https://github.com/octocat/Hello-World")
    local_id = compute_repository_id(str(tmp_path))

    assert remote_id != local_id