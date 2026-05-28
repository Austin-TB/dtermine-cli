"""Tests for the result/report schema (v2.0)."""

from __future__ import annotations

from determinism_audit.config import ConfigLabel
from determinism_audit.result import (
    ModelConfigResult,
    PromptResult,
    PromptScore,
    PromptScoreEntry,
    RunDocument,
    RunResult,
    SummaryMetrics,
)


def _make_run_result(prompt_id: str = "factual-001", run_index: int = 0) -> RunResult:
    return RunResult(
        prompt_id=prompt_id,
        run_index=run_index,
        model="openai/gpt-4o-mini",
        config_label=ConfigLabel.A,
        response="Au",
        latency_ms=123.0,
    )


def _make_prompt_score() -> PromptScore:
    return PromptScore(
        byte_exact_rate=1.0,
        byte_exact_ci_low=0.5,
        byte_exact_ci_high=1.0,
        divergence_index=0.0,
    )


def _make_summary() -> SummaryMetrics:
    return SummaryMetrics(
        byte_exact_rate=1.0,
        byte_exact_ci_low=0.5,
        byte_exact_ci_high=1.0,
        mean_divergence_index=0.0,
    )


def test_run_result_with_error() -> None:
    rr = RunResult(
        prompt_id="factual-001",
        run_index=0,
        model="openai/gpt-4o-mini",
        config_label=ConfigLabel.A,
        response=None,
        error={"code": 429, "message": "Rate limited", "type": "RateLimitError"},
        latency_ms=50.0,
    )
    assert rr.error is not None
    assert rr.response is None


def test_prompt_result_ephemeral() -> None:
    runs = [_make_run_result(run_index=i) for i in range(3)]
    pr = PromptResult(
        prompt_id="factual-001",
        category="factual",
        scoring_mode="exact",
        runs=runs,
    )
    assert len(pr.runs) == 3


def test_run_document_serialises() -> None:
    entry = PromptScoreEntry(
        prompt_id="factual-001",
        category="factual",
        scoring_mode="exact",
        score=_make_prompt_score(),
    )
    mcr = ModelConfigResult(
        model="openai/gpt-4o-mini",
        config_label=ConfigLabel.A,
        n_runs=5,
        prompt_scores=[entry],
        summary=_make_summary(),
    )
    doc = RunDocument(run_id="test-run-1", results=[mcr])
    data = doc.model_dump()
    assert data["schema_version"] == "2.0"
    assert len(data["results"]) == 1
    assert len(data["results"][0]["prompt_scores"]) == 1
    assert "runs" not in data["results"][0]["prompt_scores"][0]


def test_run_document_json_roundtrip() -> None:
    entry = PromptScoreEntry(
        prompt_id="factual-001",
        category="factual",
        scoring_mode="exact",
        score=_make_prompt_score(),
    )
    mcr = ModelConfigResult(
        model="openai/gpt-4o-mini",
        config_label=ConfigLabel.B,
        n_runs=3,
        prompt_scores=[entry],
        summary=_make_summary(),
    )
    doc = RunDocument(run_id="test-run-2", results=[mcr])
    json_str = doc.model_dump_json()
    restored = RunDocument.model_validate_json(json_str)
    assert restored.run_id == doc.run_id
    assert restored.schema_version == "2.0"
    assert restored.results[0].model == "openai/gpt-4o-mini"
    assert restored.results[0].config_label == ConfigLabel.B
