from __future__ import annotations

from pathlib import Path
from typing import List

from app.ingestion.clone import clone_repository
from app.ingestion.file_walker import walk_repository


def ingest_repository(source: str) -> List[Path]:
    """Clone a repository and return its relevant files."""
    local_repository_path = clone_repository(source)
    return walk_repository(local_repository_path)