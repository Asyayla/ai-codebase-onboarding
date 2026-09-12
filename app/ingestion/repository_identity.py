"""
repository_identity.py

Responsibility:
    Derive a stable repository_id from a clone source (anything
    clone_repository in clone.py accepts: a remote URL or a local
    filesystem path), so that equivalent references to the same
    repository resolve to the same identity.

    This module does NOT hash the result. The canonicalized string itself
    IS the repository_id - unlike chunk_id (app/embedding/chunk_identity.py),
    which combines several fields and genuinely needs a fixed-width hash,
    repository_id has exactly one input, so hashing it would only lose
    human-readability without buying anything.

    Scope, deliberately limited:
        - Remote URLs: only http/https forms are specially normalized
          (scheme casing, host casing, trailing slash, trailing ".git").
          This covers the URL forms this project actually clones from
          (see clone.py) - it is not a general-purpose git URL parser,
          and does not attempt to normalize other remote forms such as
          scp-like "git@host:path" or "ssh://"/"git://" URLs. Those fall
          through to the local-path branch below, which will produce a
          syntactically valid but likely NOT meaningfully canonicalized
          identity for them - an accepted, disclosed limitation for this
          stage, not a silently swallowed gap.
        - Local filesystem paths: resolved to an absolute, canonical path.
          This identity is only stable on the same machine - acceptable
          for now since local-path support in this project exists mainly
          for testing (see clone.py's shallow-clone-from-a-local-path
          behavior), not as a real multi-machine repository identity
          mechanism.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_NORMALIZED_REMOTE_SCHEMES = {"http", "https"}
_GIT_SUFFIX = ".git"


def compute_repository_id(source: str) -> str:
    """
    Derive a stable repository_id from `source`.

    Args:
        source: a remote URL or local filesystem path, in the same form
            clone_repository() accepts.

    Returns:
        A canonicalized string identifying the repository. Equivalent
        http(s) URL forms (differing only in scheme/host casing, a
        trailing slash, or a trailing ".git") resolve to the same string.
        Equivalent local path references (relative vs. absolute) resolve
        to the same string.
    """
    stripped = source.strip()
    parsed = urlsplit(stripped)

    if parsed.scheme.lower() in _NORMALIZED_REMOTE_SCHEMES and parsed.netloc:
        return _canonicalize_remote_url(parsed)

    return _canonicalize_local_path(stripped)


def _canonicalize_remote_url(parsed) -> str:
    scheme = parsed.scheme.lower()

    # Only the host portion of netloc is case-normalized; any userinfo
    # (rare for this project's use case - public repo URLs) is left as
    # given rather than guessing at its casing semantics.
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