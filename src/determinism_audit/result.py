"""Result schema for individual runs and output JSON structure.

Schema version 1.0 — frozen at Phase 2. Any breaking change must bump the
major version and update schemas/result.schema.json.
"""

from __future__ import annotations

import datetime
from datetime import UTC
from typing import Any

from pydantic import BaseModel, Field

from determinism_audit.config import ConfigLabel


class RunResult(BaseModel):
    """The outcome of a single prompt run."""

    prompt_id: str
    run_index: int = Field(ge=0)
    model: str
    config_label: ConfigLabel
    response: str | None = None
    error: dict[str, Any] | None = None
    latency_ms: float = Field(ge=0.0)


class PromptScore(BaseModel):
    """Aggregate scores for all runs of a single (prompt, model, config) triple."""

    byte_exact_rate: float = Field(ge=0.0, le=1.0)
    byte_exact_ci_low: float = Field(ge=0.0, le=1.0)
    byte_exact_ci_high: float = Field(ge=0.0, le=1.0)

    structural_validity_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    structural_validity_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    structural_validity_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)

    semantic_stability_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_stability_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_stability_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)

    divergence_index: float = Field(ge=0.0, le=1.0)


class PromptResult(BaseModel):
    """All runs for a single (prompt, model, config) triple."""

    prompt_id: str
    category: str
    scoring_mode: str
    runs: list[RunResult]
    score: PromptScore | None = None


class SummaryMetrics(BaseModel):
    """Aggregate metrics summarised across all prompts in the report."""

    byte_exact_rate: float = Field(ge=0.0, le=1.0)
    byte_exact_ci_low: float = Field(ge=0.0, le=1.0)
    byte_exact_ci_high: float = Field(ge=0.0, le=1.0)

    structural_validity_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    structural_validity_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    structural_validity_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)

    semantic_stability_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_stability_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_stability_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)

    mean_divergence_index: float = Field(ge=0.0, le=1.0)


class AuditReport(BaseModel):
    """Top-level output document written to disk as JSON. Schema version 1.0."""

    schema_version: str = "1.0"
    run_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    model: str
    config_label: ConfigLabel
    n_runs: int
    prompt_results: list[PromptResult]
    summary: SummaryMetrics | None = None
