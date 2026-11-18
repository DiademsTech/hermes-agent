"""Paths of a Hermes home that its hosting platform owns (``HERMES_HOST_OWNED_PATHS``).

A host can keep its own state inside the agent's home: read-only mounts of its dependencies, the
files it renders, its private working folders. A backup never carries them, and a transfer
(``hermes import --keep-host``) neither writes nor removes them: a mount cannot be written, a
rendered file is rendered again, and a host's work folder may hold the very archive being imported.

The variable lists home-relative paths separated by commas; ``*`` and ``?`` match within one path
segment (``plugins/host-*``). A path covers everything below it. Unset, nothing changes.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Iterable

ENV_VAR = "HERMES_HOST_OWNED_PATHS"


def host_owned_patterns() -> tuple[tuple[str, ...], ...]:
    """The declared paths, as segment tuples; blank, absolute or ``..`` entries are ignored."""
    patterns = []
    for raw in os.environ.get(ENV_VAR, "").split(","):
        parts = tuple(p for p in raw.strip().split("/") if p)
        if parts and not raw.strip().startswith("/") and not any(p in (".", "..") for p in parts):
            patterns.append(parts)
    return tuple(patterns)


def is_host_owned(parts: Iterable[str]) -> bool:
    """True when the home-relative path ``parts`` is, or is below, a declared path."""
    parts = tuple(parts)
    for pattern in host_owned_patterns():
        if len(parts) >= len(pattern) and all(
                fnmatch.fnmatchcase(part, segment) for part, segment in zip(parts, pattern)):
            return True
    return False
