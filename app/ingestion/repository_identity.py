from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_NORMALIZED_REMOTE_SCHEMES = {"http", "https"}
_GIT_SUFFIX = ".git"


def compute_repository_id(source: str) -> str:
    """Derive a stable repository ID from a remote URL or local path."""
    stripped = source.strip()
    parsed = urlsplit(stripped)

    if parsed.scheme.lower() in _NORMALIZED_REMOTE_SCHEMES and parsed.netloc:
        return _canonicalize_remote_url(parsed)

    return _canonicalize_local_path(stripped)


def _canonicalize_remote_url(parsed) -> str:
    scheme = parsed.scheme.lower()

    # Normalize only the host portion; userinfo may be case-sensitive.
    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, host_part = netloc.rsplit("@", 1)
        netloc = f"{userinfo}@{host_part.lower()}"
    else:
        netloc = netloc.lower()

    path = parsed.path
    if path.endswith("/"):
        path = path[:-1]
    if path.endswith(_GIT_SUFFIX):
        path = path[:-len(_GIT_SUFFIX)]

    # Query strings and fragments do not affect repository identity.
    return urlunsplit((scheme, netloc, path, "", ""))


def _canonicalize_local_path(source: str) -> str:
    return str(Path(source).expanduser().resolve())