"""Unit tests for scorers/exact.py."""

from __future__ import annotations

import pytest

from determinism_audit.scorers.exact import _normalise, all_exact, score_exact


class TestNormalise:
    def test_strips_leading_trailing_whitespace(self) -> None:
        assert _normalise("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self) -> None:
        assert _normalise("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self) -> None:
        assert _normalise("foo\t\nbar") == "foo bar"

    def test_empty_string(self) -> None:
        assert _normalise("") == ""

    def test_all_whitespace(self) -> None:
        assert _normalise("   \t\n  ") == ""


class TestScoreExact:
    def test_identical_strings_match(self) -> None:
        assert score_exact("Au", "Au") is True

    def test_empty_strings_match(self) -> None:
        assert score_exact("", "") is True

    def test_different_strings_do_not_match(self) -> None:
        assert score_exact("Au", "Ag") is False

    def test_normalisation_applied(self) -> None:
        assert score_exact("  Au  ", "Au") is True

    def test_internal_whitespace_normalised(self) -> None:
        assert score_exact("hello   world", "hello world") is True

    def test_case_sensitive(self) -> None:
        assert score_exact("au", "Au") is False

    def test_trailing_newline_ignored(self) -> None:
        assert score_exact("Au\n", "Au") is True


class TestAllExact:
    def test_empty_list(self) -> None:
        assert all_exact([]) is True

    def test_single_element(self) -> None:
        assert all_exact(["Au"]) is True

    def test_all_same(self) -> None:
        assert all_exact(["Au", "Au", "Au"]) is True

    def test_one_differs(self) -> None:
        assert all_exact(["Au", "Au", "Ag"]) is False

    def test_normalisation_applied_across_all(self) -> None:
        assert all_exact(["Au ", " Au", "Au"]) is True

    def test_empty_strings_all_same(self) -> None:
        assert all_exact(["", "", ""]) is True

    def test_mixed_empty_and_nonempty(self) -> None:
        assert all_exact(["", "Au"]) is False
