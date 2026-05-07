"""Unit tests for scorers/structural.py."""

from __future__ import annotations

import pytest

from determinism_audit.scorers.structural import (
    _extract_code,
    _extract_json,
    _key_type_signature,
    all_structurally_valid,
    json_structure_match,
    score_json,
    score_python_ast,
    score_structural,
)


class TestExtractJson:
    def test_plain_json_unchanged(self) -> None:
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_strips_json_fence(self) -> None:
        result = _extract_json("```json\n{\"a\": 1}\n```")
        assert result == '{"a": 1}'

    def test_strips_plain_fence(self) -> None:
        result = _extract_json("```\n{\"a\": 1}\n```")
        assert result == '{"a": 1}'

    def test_strips_surrounding_whitespace(self) -> None:
        assert _extract_json("  [1, 2]  ") == "[1, 2]"


class TestExtractCode:
    def test_plain_code_unchanged(self) -> None:
        assert _extract_code("x = 1") == "x = 1"

    def test_strips_python_fence(self) -> None:
        result = _extract_code("```python\nx = 1\n```")
        assert result == "x = 1"


class TestKeyTypeSignature:
    def test_dict_structure(self) -> None:
        sig = _key_type_signature({"a": 1, "b": "hello"})
        assert sig == {"a": "int", "b": "str"}

    def test_list_structure(self) -> None:
        sig = _key_type_signature([1, 2, 3])
        assert sig == ["list", ["int"]]

    def test_nested_dict(self) -> None:
        sig = _key_type_signature({"x": {"y": True}})
        assert sig == {"x": {"y": "bool"}}

    def test_primitive(self) -> None:
        assert _key_type_signature(42) == "int"
        assert _key_type_signature("hello") == "str"
        assert _key_type_signature(3.14) == "float"

    def test_mixed_list(self) -> None:
        sig = _key_type_signature([1, "a"])
        assert sig == ["list", ["int", "str"]]


class TestScoreJson:
    def test_valid_object(self) -> None:
        assert score_json('{"key": "value"}') is True

    def test_valid_array(self) -> None:
        assert score_json("[1, 2, 3]") is True

    def test_invalid_json(self) -> None:
        assert score_json("not json") is False

    def test_empty_string(self) -> None:
        assert score_json("") is False

    def test_fenced_json(self) -> None:
        assert score_json("```json\n{\"a\": 1}\n```") is True

    def test_null_json(self) -> None:
        assert score_json("null") is True


class TestJsonStructureMatch:
    def test_same_structure(self) -> None:
        assert json_structure_match('{"a": 1}', '{"a": 2}') is True

    def test_different_keys(self) -> None:
        assert json_structure_match('{"a": 1}', '{"b": 1}') is False

    def test_different_types(self) -> None:
        assert json_structure_match('{"a": 1}', '{"a": "x"}') is False

    def test_both_invalid_json(self) -> None:
        assert json_structure_match("bad", "also bad") is False

    def test_one_invalid(self) -> None:
        assert json_structure_match('{"a": 1}', "bad") is False

    def test_arrays_same_element_type(self) -> None:
        assert json_structure_match("[1, 2]", "[3, 4]") is True

    def test_arrays_different_element_type(self) -> None:
        assert json_structure_match("[1, 2]", '["a", "b"]') is False


class TestScorePythonAst:
    def test_valid_function(self) -> None:
        code = "def add(a, b):\n    return a + b"
        assert score_python_ast(code) is True

    def test_valid_expression(self) -> None:
        assert score_python_ast("x = [i**2 for i in range(10)]") is True

    def test_invalid_syntax(self) -> None:
        assert score_python_ast("def broken(") is False

    def test_fenced_code(self) -> None:
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        assert score_python_ast(code) is True

    def test_empty_string(self) -> None:
        assert score_python_ast("") is True  # empty is valid Python


class TestScoreStructural:
    def test_json_path_same_structure(self) -> None:
        assert score_structural('{"a": 1}', '{"a": 99}') is True

    def test_json_path_different_structure(self) -> None:
        assert score_structural('{"a": 1}', '{"b": 1}') is False

    def test_python_path_both_valid(self) -> None:
        a = "def foo():\n    return 1"
        b = "def bar():\n    return 2"
        assert score_structural(a, b) is True

    def test_python_path_one_invalid(self) -> None:
        a = "def foo():\n    return 1"
        b = "def broken("
        assert score_structural(a, b) is False

    def test_neither_json_nor_python(self) -> None:
        assert score_structural("not code", "also not code") is False


class TestAllStructurallyValid:
    def test_all_valid_json(self) -> None:
        result = all_structurally_valid(['{"a": 1}', "[1, 2]", "null"])
        assert result == [True, True, True]

    def test_all_valid_python(self) -> None:
        result = all_structurally_valid(["x = 1", "def f(): pass"])
        assert result == [True, True]

    def test_mixed_valid_invalid(self) -> None:
        result = all_structurally_valid(['{"a": 1}', "not valid at all $$$$"])
        assert result[0] is True
        assert result[1] is False

    def test_empty_list(self) -> None:
        assert all_structurally_valid([]) == []
