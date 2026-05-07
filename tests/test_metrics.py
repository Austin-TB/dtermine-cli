"""Unit tests for metrics.py."""

from __future__ import annotations

import json
import math
from unittest.mock import patch

import pytest

from determinism_audit.config import ConfigLabel
from determinism_audit.metrics import (
    _levenshtein,
    _normalised_levenshtein,
    _wilson_ci,
    byte_exact_rate,
    divergence_index,
    drift_delta,
    semantic_stability_rate,
    structural_validity_rate,
)
from determinism_audit.result import RunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(response: str | None = "Au", run_index: int = 0) -> RunResult:
    return RunResult(
        prompt_id="test-001",
        run_index=run_index,
        model="openai/gpt-4o-mini",
        config_label=ConfigLabel.A,
        response=response,
        latency_ms=10.0,
    )


def _error_run(run_index: int = 0) -> RunResult:
    return RunResult(
        prompt_id="test-001",
        run_index=run_index,
        model="openai/gpt-4o-mini",
        config_label=ConfigLabel.A,
        response=None,
        error={"code": 429, "message": "rate limited", "type": "RateLimitError"},
        latency_ms=5.0,
    )


# ---------------------------------------------------------------------------
# Test _wilson_ci
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_all_successes(self) -> None:
        rate, lo, hi = _wilson_ci(5, 5)
        assert rate == pytest.approx(1.0)
        assert lo > 0.5
        assert hi == pytest.approx(1.0)

    def test_no_successes(self) -> None:
        rate, lo, hi = _wilson_ci(0, 5)
        assert rate == pytest.approx(0.0)
        assert lo == pytest.approx(0.0)
        assert hi < 0.5

    def test_half_successes(self) -> None:
        rate, lo, hi = _wilson_ci(5, 10)
        assert rate == pytest.approx(0.5)
        assert lo < 0.5
        assert hi > 0.5

    def test_zero_trials(self) -> None:
        rate, lo, hi = _wilson_ci(0, 0)
        assert rate == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_ci_bounds_in_zero_one(self) -> None:
        rate, lo, hi = _wilson_ci(3, 4)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0


# ---------------------------------------------------------------------------
# Test Levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical_strings(self) -> None:
        assert _levenshtein("abc", "abc") == 0

    def test_empty_to_nonempty(self) -> None:
        assert _levenshtein("", "abc") == 3

    def test_nonempty_to_empty(self) -> None:
        assert _levenshtein("abc", "") == 3

    def test_single_substitution(self) -> None:
        assert _levenshtein("abc", "axc") == 1

    def test_single_insertion(self) -> None:
        assert _levenshtein("ab", "abc") == 1

    def test_single_deletion(self) -> None:
        assert _levenshtein("abc", "ab") == 1

    def test_completely_different(self) -> None:
        assert _levenshtein("abc", "xyz") == 3

    def test_both_empty(self) -> None:
        assert _levenshtein("", "") == 0


class TestNormalisedLevenshtein:
    def test_identical_strings_zero(self) -> None:
        assert _normalised_levenshtein("abc", "abc") == pytest.approx(0.0)

    def test_completely_different_approaches_one(self) -> None:
        dist = _normalised_levenshtein("abc", "xyz")
        assert dist == pytest.approx(1.0)

    def test_both_empty(self) -> None:
        assert _normalised_levenshtein("", "") == pytest.approx(0.0)

    def test_empty_and_nonempty(self) -> None:
        assert _normalised_levenshtein("", "abc") == pytest.approx(1.0)

    def test_range_zero_to_one(self) -> None:
        d = _normalised_levenshtein("hello world", "hello")
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# Test byte_exact_rate
# ---------------------------------------------------------------------------

class TestByteExactRate:
    def test_all_identical(self) -> None:
        runs = [_run("Au", i) for i in range(5)]
        rate, lo, hi = byte_exact_rate(runs)
        assert rate == pytest.approx(1.0)

    def test_all_different(self) -> None:
        runs = [
            _run("Au", 0), _run("Ag", 1), _run("Cu", 2),
            _run("Fe", 3), _run("Au", 4),
        ]
        rate, lo, hi = byte_exact_rate(runs)
        # Only 2 of 5 match the pivot ("Au")
        assert rate == pytest.approx(2 / 5)

    def test_error_run_counts_as_mismatch(self) -> None:
        runs = [_run("Au", 0), _run("Au", 1), _error_run(2)]
        rate, lo, hi = byte_exact_rate(runs)
        assert rate == pytest.approx(2 / 3)

    def test_empty_runs(self) -> None:
        rate, lo, hi = byte_exact_rate([])
        assert rate == 0.0

    def test_all_error_runs(self) -> None:
        runs = [_error_run(i) for i in range(3)]
        rate, lo, hi = byte_exact_rate(runs)
        assert rate == 0.0

    def test_normalisation_applied(self) -> None:
        runs = [_run("Au ", 0), _run(" Au", 1), _run("Au", 2)]
        rate, lo, hi = byte_exact_rate(runs)
        assert rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test semantic_stability_rate
# ---------------------------------------------------------------------------

class TestSemanticStabilityRate:
    def test_fewer_than_two_responses(self) -> None:
        runs = [_run("hello", 0)]
        rate, lo, hi = semantic_stability_rate(runs)
        assert rate == 0.0 and lo == 0.0 and hi == 0.0

    def test_empty_runs(self) -> None:
        rate, lo, hi = semantic_stability_rate([])
        assert rate == 0.0

    def test_all_similar(self) -> None:
        with patch("determinism_audit.metrics.pairwise_similarities") as mock_sim:
            mock_sim.return_value = [1.0, 1.0, 1.0]
            runs = [_run("a", i) for i in range(3)]
            rate, lo, hi = semantic_stability_rate(runs)
            assert rate == pytest.approx(1.0)

    def test_none_similar(self) -> None:
        with patch("determinism_audit.metrics.pairwise_similarities") as mock_sim:
            mock_sim.return_value = [0.0, 0.0, 0.0]
            runs = [_run("a", i) for i in range(3)]
            rate, lo, hi = semantic_stability_rate(runs)
            assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test structural_validity_rate
# ---------------------------------------------------------------------------

class TestStructuralValidityRate:
    def test_all_valid_json(self) -> None:
        runs = [_run('{"a": 1}', i) for i in range(3)]
        rate, lo, hi = structural_validity_rate(runs)
        assert rate == pytest.approx(1.0)

    def test_all_invalid(self) -> None:
        runs = [_run("not valid $$$", i) for i in range(3)]
        rate, lo, hi = structural_validity_rate(runs)
        assert rate == pytest.approx(0.0)

    def test_error_run_invalid(self) -> None:
        runs = [_run('{"a": 1}', 0), _error_run(1)]
        rate, lo, hi = structural_validity_rate(runs)
        assert rate == pytest.approx(0.5)

    def test_valid_python_counts(self) -> None:
        runs = [_run("def foo(): pass", i) for i in range(2)]
        rate, lo, hi = structural_validity_rate(runs)
        assert rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test divergence_index
# ---------------------------------------------------------------------------

class TestDivergenceIndex:
    def test_identical_responses_zero(self) -> None:
        runs = [_run("Au", i) for i in range(5)]
        di = divergence_index(runs)
        assert di == pytest.approx(0.0)

    def test_fewer_than_two_runs(self) -> None:
        assert divergence_index([]) == 0.0
        assert divergence_index([_run("Au")]) == 0.0

    def test_error_runs_treated_as_empty(self) -> None:
        runs = [_run("hello", 0), _error_run(1)]
        di = divergence_index(runs)
        assert 0.0 <= di <= 1.0

    def test_completely_different_high_di(self) -> None:
        runs = [_run("a" * 100, 0), _run("b" * 100, 1)]
        di = divergence_index(runs)
        assert di > 0.9

    def test_range_zero_to_one(self) -> None:
        runs = [_run("hello world", i) for i in range(3)]
        di = divergence_index(runs)
        assert 0.0 <= di <= 1.0


# ---------------------------------------------------------------------------
# Test drift_delta
# ---------------------------------------------------------------------------

class TestDriftDelta:
    def _make_doc(self, model: str, ber: float) -> dict:  # type: ignore[type-arg]
        return {
            "model": model,
            "prompt_results": [
                {
                    "category": "factual",
                    "score": {"byte_exact_rate": ber},
                }
            ],
        }

    def test_positive_delta(self) -> None:
        doc_a = self._make_doc("openai/gpt-4o-mini", 0.6)
        doc_b = self._make_doc("openai/gpt-4o-mini", 0.8)
        deltas = drift_delta(doc_a, doc_b)
        key = ("openai/gpt-4o-mini", "factual")
        assert deltas[key] == pytest.approx(0.2)

    def test_negative_delta(self) -> None:
        doc_a = self._make_doc("openai/gpt-4o-mini", 0.9)
        doc_b = self._make_doc("openai/gpt-4o-mini", 0.5)
        deltas = drift_delta(doc_a, doc_b)
        key = ("openai/gpt-4o-mini", "factual")
        assert deltas[key] == pytest.approx(-0.4)

    def test_key_only_in_b(self) -> None:
        doc_a: dict = {"model": "m", "prompt_results": []}  # type: ignore[type-arg]
        doc_b = self._make_doc("m", 0.7)
        deltas = drift_delta(doc_a, doc_b)
        assert deltas[("m", "factual")] == pytest.approx(0.7)

    def test_key_only_in_a(self) -> None:
        doc_a = self._make_doc("m", 0.7)
        doc_b: dict = {"model": "m", "prompt_results": []}  # type: ignore[type-arg]
        deltas = drift_delta(doc_a, doc_b)
        assert deltas[("m", "factual")] == pytest.approx(-0.7)

    def test_load_from_file(self, tmp_path: pytest.TempPathFactory) -> None:
        doc_a = self._make_doc("m", 0.5)
        doc_b = self._make_doc("m", 0.9)
        path_a = tmp_path / "a.json"  # type: ignore[operator]
        path_b = tmp_path / "b.json"  # type: ignore[operator]
        path_a.write_text(json.dumps(doc_a))
        path_b.write_text(json.dumps(doc_b))
        deltas = drift_delta(str(path_a), str(path_b))
        assert deltas[("m", "factual")] == pytest.approx(0.4)
