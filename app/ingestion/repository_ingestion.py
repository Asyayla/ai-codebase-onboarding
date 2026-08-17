"""
repository_ingestion.py

Responsibility:
    Compose clone_repository (clone.py) and walk_repository (file_walker.py)
    into the single end-to-end operation this project actually needs:
    "given a source, tell me which files in it are worth analyzing."

    This module contains no filesystem-filtering logic and no cloning logic
    of its own - both already exist, are already tested, and are reused here
    unchanged. This module is intentionally just sequencing.

Contract:
    ingest_repository(source: str) -> List[Path]

    - Clones `source` via clone_repository (raises RepositoryCloneError on
      failure - not caught or altered here).
    - Walks the resulting local clone via walk_repository.
    - Returns exactly what walk_repository returns: a sorted list of Path
      objects, relative to the cloned repository's root.
    - Does NOT delete the cloned directory. Cleanup is out of scope for this
      module, matching clone_repository's own current behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from app.ingestion.clone import clone_repository
from app.ingestion.file_walker import walk_repository


def ingest_repository(source: str) -> List[Path]:
    """
    Clone `source` and return the list of relevant files found in it.

    Args:
        source: anything clone_repository accepts - a remote URL or a local
            path that git can clone from.

    Returns:
        Sorted list of Path objects, relative to the cloned repository root,
        as produced by walk_repository.

    Raises:
        RepositoryCloneError: if cloning `source` fails. Propagated
            unchanged from clone_repository.
    """
    local_repository_path = clone_repository(source)
    return walk_repository(local_repository_path)