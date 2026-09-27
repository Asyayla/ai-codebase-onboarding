from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class RepositoryCloneError(Exception):
    """Raised when git clone fails."""


def clone_repository(source: str) -> Path:
    """Shallow-clone a Git repository into a temporary directory."""
    destination = Path(tempfile.mkdtemp(prefix="repo_clone_"))

    # --no-local keeps --depth effective for local repository sources too.
    command = [
        "git",
        "clone",
        "--depth",
        "1",
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