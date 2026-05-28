"""Exact-match scorer: whitespace-normalise then byte-compare."""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends."""
    return _WS_RE.sub(" ", text).strip()


def score_exact(a: str, b: str) -> bool:
    """Return True iff *a* and *b* are identical after whitespace normalisation."""
    return _normalise(a) == _normalise(b)


def all_exact(responses: list[str]) -> bool:
    """Return True iff every response in *responses* is identical to the first."""
    if not responses:
        return True
    normalised = [_normalise(r) for r in responses]
    pivot = normalised[0]
    return all(r == pivot for r in normalised[1:])
