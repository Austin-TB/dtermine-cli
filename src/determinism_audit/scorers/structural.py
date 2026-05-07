"""Structural scorer: JSON validity/shape check and Python AST parse check."""

from __future__ import annotations

import ast
import json
import re

# ---------------------------------------------------------------------------
# JSON structural scoring
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Strip markdown code fences if present, otherwise return text as-is."""
    # Strip ```json ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _key_type_signature(obj: object) -> object:
    """Recursively reduce a parsed JSON value to its structural shape."""
    if isinstance(obj, dict):
        return {k: _key_type_signature(v) for k, v in obj.items()}
    if isinstance(obj, list):
        element_types = {type(item).__name__ for item in obj}
        return ["list", sorted(element_types)]
    return type(obj).__name__


def score_json(text: str) -> bool:
    """Return True iff *text* (after stripping code fences) is valid JSON."""
    try:
        json.loads(_extract_json(text))
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def json_structure_match(a: str, b: str) -> bool:
    """Return True iff *a* and *b* parse to the same structural shape."""
    try:
        obj_a = json.loads(_extract_json(a))
        obj_b = json.loads(_extract_json(b))
    except (json.JSONDecodeError, ValueError):
        return False
    return _key_type_signature(obj_a) == _key_type_signature(obj_b)


# ---------------------------------------------------------------------------
# Python code structural scoring
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```")


def _extract_code(text: str) -> str:
    """Strip markdown Python code fences if present."""
    fenced = _CODE_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def score_python_ast(text: str) -> bool:
    """Return True iff *text* (after stripping fences) is valid Python."""
    try:
        ast.parse(_extract_code(text))
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Unified structural scorer
# ---------------------------------------------------------------------------

def score_structural(a: str, b: str) -> bool:
    """
    Return True iff *a* and *b* are structurally equivalent.

    - If both parse as JSON, compare structural shape.
    - Otherwise, both must parse as valid Python AST.
    """
    # Try JSON path first
    a_cleaned = _extract_json(a)
    b_cleaned = _extract_json(b)
    try:
        obj_a = json.loads(a_cleaned)
        obj_b = json.loads(b_cleaned)
        return _key_type_signature(obj_a) == _key_type_signature(obj_b)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to Python AST validity check
    return score_python_ast(a) and score_python_ast(b)


def all_structurally_valid(responses: list[str]) -> list[bool]:
    """Return a per-response boolean indicating structural validity."""
    results: list[bool] = []
    for r in responses:
        # Valid if it's valid JSON or valid Python
        if score_json(r):
            results.append(True)
        elif score_python_ast(r):
            results.append(True)
        else:
            results.append(False)
    return results
