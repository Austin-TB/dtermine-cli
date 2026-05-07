"""Unit tests for scorers/semantic.py.

These tests mock the embedding model to avoid network calls and heavy
dependencies in CI, while still exercising all code paths.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from determinism_audit.scorers.semantic import (
    DEFAULT_THRESHOLD,
    _cosine,
    all_above_threshold,
    encode,
    pairwise_similarities,
    score_semantic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(n: int, idx: int) -> list[float]:
    """Return a unit vector of length n with 1.0 at position idx."""
    v = [0.0] * n
    v[idx] = 1.0
    return v


def _identical_embeddings(texts: list[str]) -> list[list[float]]:
    """Return identical unit vectors for all texts (similarity == 1.0)."""
    return [_unit_vec(4, 0) for _ in texts]


def _orthogonal_embeddings(texts: list[str]) -> list[list[float]]:
    """Return orthogonal unit vectors (similarity == 0.0)."""
    return [_unit_vec(4, i % 4) for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Test _cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_vector_b(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_general_case(self) -> None:
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test encode (mocked)
# ---------------------------------------------------------------------------

class TestEncode:
    def test_returns_list_of_float_lists(self) -> None:
        with patch("determinism_audit.scorers.semantic._load_model") as mock_load:
            mock_model = MagicMock()
            import numpy as np
            mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_load.return_value = mock_model

            result = encode(["a", "b"])
            assert len(result) == 2
            assert all(isinstance(v, list) for v in result)
            assert all(isinstance(x, float) for x in result[0])


# ---------------------------------------------------------------------------
# Test score_semantic (mocked)
# ---------------------------------------------------------------------------

class TestScoreSemantic:
    def test_identical_texts_above_threshold(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_identical_embeddings):
            assert score_semantic("hello", "hello") is True

    def test_orthogonal_texts_below_threshold(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_orthogonal_embeddings):
            assert score_semantic("hello", "world") is False

    def test_custom_threshold_zero_always_passes(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_orthogonal_embeddings):
            assert score_semantic("hello", "world", threshold=0.0) is True

    def test_custom_threshold_one_only_identical(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_identical_embeddings):
            assert score_semantic("hello", "hello", threshold=1.0) is True


# ---------------------------------------------------------------------------
# Test pairwise_similarities (mocked)
# ---------------------------------------------------------------------------

class TestPairwiseSimilarities:
    def test_empty_list(self) -> None:
        assert pairwise_similarities([]) == []

    def test_single_text(self) -> None:
        assert pairwise_similarities(["hello"]) == []

    def test_two_identical_texts(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_identical_embeddings):
            sims = pairwise_similarities(["a", "b"])
            assert len(sims) == 1
            assert sims[0] == pytest.approx(1.0)

    def test_three_orthogonal_texts(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_orthogonal_embeddings):
            sims = pairwise_similarities(["a", "b", "c"])
            assert len(sims) == 3  # (0,1), (0,2), (1,2)
            assert all(s == pytest.approx(0.0) for s in sims)

    def test_five_texts_yields_ten_pairs(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_identical_embeddings):
            sims = pairwise_similarities(["a", "b", "c", "d", "e"])
            assert len(sims) == 10  # C(5, 2)


# ---------------------------------------------------------------------------
# Test all_above_threshold (mocked)
# ---------------------------------------------------------------------------

class TestAllAboveThreshold:
    def test_empty_list(self) -> None:
        assert all_above_threshold([]) is True

    def test_single_text(self) -> None:
        assert all_above_threshold(["hello"]) is True

    def test_all_identical_above_threshold(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_identical_embeddings):
            assert all_above_threshold(["a", "b", "c"]) is True

    def test_orthogonal_below_threshold(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_orthogonal_embeddings):
            assert all_above_threshold(["a", "b"], threshold=0.5) is False

    def test_zero_threshold_always_true(self) -> None:
        with patch("determinism_audit.scorers.semantic.encode", side_effect=_orthogonal_embeddings):
            assert all_above_threshold(["a", "b"], threshold=0.0) is True
