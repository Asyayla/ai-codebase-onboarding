"""
clone.py

Responsibility:
    Given a source that git understands (a remote URL or a local filesystem
    path to a git repository), produce a shallow local clone in a freshly
    created temporary directory and return its path.

    This module does NOT validate the source (e.g. restricting to
    github.com), does NOT walk the resulting file tree, and does NOT manage
    the lifecycle/retention of the returned directory beyond removing it if
    the clone itself fails partway through. Cleanup of successful clones is
    the responsibility of a later lifecycle layer, not this module.

Contract:
    clone_repository(source: str) -> Path

    - Creates a new temporary directory via tempfile.mkdtemp.
    - Runs git clone --depth 1 --no-tags --no-local <source> <destination>
      using subprocess.run with an explicit argument list (never shell=True).
    - On success, returns the destination Path. The directory is left in
      place; nothing in this module deletes a successful clone.
    - On failure (non-zero git exit code, or git executable not found),
      raises RepositoryCloneError with the underlying failure details, and
      removes the (empty or partial) temporary directory it created.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class RepositoryCloneError(Exception):
    """Raised when git clone fails for any reason: an invalid or
    unreachable source, a non-zero git exit code, or the git executable
    not being available on the system."""


def clone_repository(source: str) -> Path:
    """
    Shallow-clone `source` (a URL or local path that git can clone from)
    into a new temporary directory and return that directory's Path.

    Args:
        source: anything git clone accepts as a source - a remote URL or
            a local filesystem path.

    Returns:
        Path to the temporary directory containing the shallow clone.

    Raises:
        RepositoryCloneError: if the clone fails for any reason.
    """
    destination = Path(tempfile.mkdtemp(prefix="repo_clone_"))

    # --no-local forces git to honor --depth even when `source` happens to be
    # a local filesystem path (git silently ignores --depth for local clones
    # otherwise, only emitting a warning). This keeps shallow-clone behavior
    # identical whether `source` is a remote URL or a local path, which
    # matters both for correctness and for testability against local fixture
    # repositories.
    command = [
        "git", "clone",
        "--depth", "1",
        "--no-tags",
        "--no-local",
        source,
        str(destination),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
    except FileNotFoundError as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise RepositoryCloneError(
            "git executable not found on this system. "
            "Ensure git is installed and available on PATH."
        ) from exc

    if result.returncode != 0:
        shutil.rmtree(destination, ignore_errors=True)
        raise RepositoryCloneError(
            f"git clone failed for source '{source}' "
            f"(exit code {result.returncode}).\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )

    return destination