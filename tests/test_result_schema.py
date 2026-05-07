"""Tests for the result/report schema."""

from __future__ import annotations

from determinism_audit.config import ConfigLabel
from determinism_audit.result import AuditReport, PromptResult, RunResult


def _make_run_result(prompt_id: str = "factual-001", run_index: int = 0) -> RunResult:
    return RunResult(
        prompt_id=prompt_id,
        run_index=run_index,
        model="openai/gpt-4o-mini",
        config_label=ConfigLabel.A,
        response="Au",
        latency_ms=123.0,
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


def test_audit_report_serialises() -> None:
    runs = [_make_run_result(run_index=i) for i in range(5)]
    pr = PromptResult(
        prompt_id="factual-001",
        category="factual",
        scoring_mode="exact",
        runs=runs,
    )
    report = AuditReport(
        run_id="test-run-1",
        model="openai/gpt-4o-mini",
        config_label=ConfigLabel.A,
        n_runs=5,
        prompt_results=[pr],
    )
    data = report.model_dump()
    assert data["schema_version"] == "1.0"
    assert len(data["prompt_results"]) == 1
    assert len(data["prompt_results"][0]["runs"]) == 5


def test_audit_report_json_roundtrip() -> None:
    runs = [_make_run_result(run_index=i) for i in range(3)]
    pr = PromptResult(
        prompt_id="factual-001",
        category="factual",
        scoring_mode="exact",
        runs=runs,
    )
    report = AuditReport(
        run_id="test-run-2",
        model="openai/gpt-4o-mini",
        config_label=ConfigLabel.B,
        n_runs=3,
        prompt_results=[pr],
    )
    json_str = report.model_dump_json()
    restored = AuditReport.model_validate_json(json_str)
    assert restored.run_id == report.run_id
    assert restored.model == report.model
    assert restored.schema_version == "1.0"
