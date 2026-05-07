"""Tests for the canary prompt loader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from determinism_audit.canary.loader import load_prompts
from determinism_audit.canary.schema import Prompt


def test_load_default_prompts() -> None:
    prompts = load_prompts()
    assert len(prompts) == 100
    for p in prompts:
        assert isinstance(p, Prompt)
        assert p.id
        assert p.prompt
        assert p.max_tokens > 0


def test_all_categories_present() -> None:
    prompts = load_prompts()
    categories = {p.category for p in prompts}
    expected = {
        "factual", "structured_json", "code", "tool_calls",
        "math", "format_compliance", "longform_summary", "ambiguous_open",
    }
    assert expected == categories


def test_category_counts() -> None:
    prompts = load_prompts()
    from collections import Counter
    counts = Counter(p.category for p in prompts)
    assert counts["factual"] == 13
    assert counts["structured_json"] == 13
    assert counts["code"] == 13
    assert counts["tool_calls"] == 12
    assert counts["math"] == 12
    assert counts["format_compliance"] == 12
    assert counts["longform_summary"] == 13
    assert counts["ambiguous_open"] == 12


def test_ids_are_unique() -> None:
    prompts = load_prompts()
    ids = [p.id for p in prompts]
    assert len(ids) == len(set(ids))


def test_json_schema_present_for_structured_json() -> None:
    prompts = load_prompts()
    structured = [p for p in prompts if p.category == "structured_json"]
    for p in structured:
        assert p.json_schema is not None, f"{p.id} missing json_schema"


def test_load_custom_file() -> None:
    entry = {
        "id": "test-001",
        "category": "factual",
        "prompt": "What is 1+1?",
        "scoring_mode": "exact",
        "max_tokens": 10,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(entry) + "\n")
        tmp_path = Path(f.name)

    prompts = load_prompts(tmp_path)
    assert len(prompts) == 1
    assert prompts[0].id == "test-001"

    tmp_path.unlink()


def test_invalid_json_raises() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("not-valid-json\n")
        tmp_path = Path(f.name)

    with pytest.raises(ValueError, match="invalid JSON"):
        load_prompts(tmp_path)

    tmp_path.unlink()


def test_schema_violation_raises() -> None:
    bad_entry = {"id": "x", "category": "factual"}  # missing required fields
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(bad_entry) + "\n")
        tmp_path = Path(f.name)

    with pytest.raises(ValueError, match="schema validation failed"):
        load_prompts(tmp_path)

    tmp_path.unlink()
