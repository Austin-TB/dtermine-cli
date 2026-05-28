"""Aggregate metrics computed over a set of RunResults.

All rate functions return (rate, ci_low, ci_high) using the Wilson 95% CI.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from determinism_audit.scorers.exact import _normalise
from determinism_audit.scorers.semantic import DEFAULT_THRESHOLD, pairwise_similarities
from determinism_audit.scorers.structural import all_structurally_valid

if TYPE_CHECKING:
    from determinism_audit.result import RunResult

# ---------------------------------------------------------------------------
# Wilson score interval (95 %)
# ---------------------------------------------------------------------------

_Z95 = 1.96  # z-score for 95 % confidence


def _wilson_ci(k: int, n: int) -> tuple[float, float, float]:
    """
    Return (rate, ci_low, ci_high) using the Wilson score interval.

    *k* is the number of successes, *n* the total number of trials.
    Returns (0.0, 0.0, 0.0) when n == 0.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (_Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# Levenshtein distance (Wagner-Fischer, O(n) space)
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _normalised_levenshtein(a: str, b: str) -> float:
    """Return Levenshtein distance normalised to [0, 1]."""
    dist = _levenshtein(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return dist / max_len


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def byte_exact_rate(runs: list[RunResult]) -> tuple[float, float, float]:
    """
    Fraction of runs whose (whitespace-normalised) response matches the first
    non-error response.

    Returns (rate, ci_low, ci_high).  Errored runs count as mismatches.
    """
    responses = [r.response for r in runs if r.response is not None]
    if not responses:
        return 0.0, 0.0, 0.0
    pivot = _normalise(responses[0])
    # Count all runs (including errors) against total
    total = len(runs)
    matches = sum(1 for r in runs if r.response is not None and _normalise(r.response) == pivot)
    return _wilson_ci(matches, total)


def semantic_stability_rate(
    runs: list[RunResult],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[float, float, float]:
    """
    Fraction of run-pairs whose cosine similarity >= *threshold*.

    Returns (rate, ci_low, ci_high).  Returns (0, 0, 0) if fewer than 2
    valid responses are available (no pairs to compare).
    """
    responses = [r.response for r in runs if r.response is not None]
    if len(responses) < 2:
        return 0.0, 0.0, 0.0
    sims = pairwise_similarities(responses)
    k = sum(1 for s in sims if s >= threshold)
    return _wilson_ci(k, len(sims))


def structural_validity_rate(runs: list[RunResult]) -> tuple[float, float, float]:
    """
    Fraction of runs that produce structurally valid output (valid JSON or
    valid Python AST).

    Returns (rate, ci_low, ci_high).  Errored runs count as invalid.
    """
    if not runs:
        return 0.0, 0.0, 0.0
    validity: list[bool] = []
    for r in runs:
        if r.response is None:
            validity.append(False)
        else:
            validity.extend(all_structurally_valid([r.response]))
    k = sum(validity)
    return _wilson_ci(k, len(runs))


def divergence_index(runs: list[RunResult]) -> float:
    """
    Mean normalised Levenshtein distance over the most-divergent pair, range 0-1.

    Uses whitespace-normalised responses; errored runs are treated as empty
    strings.  Returns 0.0 when fewer than 2 runs are available.
    """
    texts = [_normalise(r.response) if r.response is not None else "" for r in runs]
    if len(texts) < 2:
        return 0.0
    max_dist = 0.0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            d = _normalised_levenshtein(texts[i], texts[j])
            if d > max_dist:
                max_dist = d
    return max_dist


def drift_delta(
    json_a: str | dict,  # type: ignore[type-arg]
    json_b: str | dict,  # type: ignore[type-arg]
) -> dict[tuple[str, str], float]:
    """
    Per-(model, category) byte-exact-rate delta between two audit reports.

    *json_a* and *json_b* may be file paths (str) or already-parsed dicts.
    Returns a mapping of ``(model, category) -> ber_b - ber_a``.
    A positive delta means the model became *more* deterministic.
    """

    def _load(src: str | dict) -> dict:  # type: ignore[type-arg]
        if isinstance(src, str):
            with open(src, encoding="utf-8") as fh:
                return json.load(fh)  # type: ignore[no-any-return]
        return src

    doc_a = _load(json_a)
    doc_b = _load(json_b)

    def _ber_map(doc: dict) -> dict[tuple[str, str], float]:  # type: ignore[type-arg]
        model = doc.get("model", "")
        result: dict[tuple[str, str], float] = {}
        for pr in doc.get("prompt_results", []):
            category = pr.get("category", "")
            score = pr.get("score")
            if score is not None:
                result[(model, category)] = score.get("byte_exact_rate", 0.0)
        return result

    map_a = _ber_map(doc_a)
    map_b = _ber_map(doc_b)

    deltas: dict[tuple[str, str], float] = {}
    all_keys = set(map_a) | set(map_b)
    for key in all_keys:
        deltas[key] = map_b.get(key, 0.0) - map_a.get(key, 0.0)
    return deltas
