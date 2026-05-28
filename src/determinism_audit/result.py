"""Result schema for the determinism audit.

Schema version 2.0:
- One RunDocument per invocation (all models times configs in one file).
- Raw LLM responses not stored; only PromptScore per prompt.
- RunResult and PromptResult are ephemeral types used during scoring only.
"""

from __future__ import annotations

import datetime
from datetime import UTC
from typing import Any

from pydantic import BaseModel, Field

from determinism_audit.config import ConfigLabel

# ---------------------------------------------------------------------------
# Ephemeral types — used during a run, never serialised to disk
# ---------------------------------------------------------------------------


class RunResult(BaseModel):
    """Outcome of a single prompt invocation. Not stored on disk."""

    prompt_id: str
    run_index: int = Field(ge=0)
    model: str
    config_label: ConfigLabel
    response: str | None = None
    error: dict[str, Any] | None = None
    latency_ms: float = Field(ge=0.0)


class PromptResult(BaseModel):
    """All runs for a (prompt, model, config) triple. Not stored on disk."""

    prompt_id: str
    category: str
    scoring_mode: str
    runs: list[RunResult]


# ---------------------------------------------------------------------------
# Stored schema — serialised to disk as RunDocument (schema version 2.0)
# ---------------------------------------------------------------------------


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


class PromptScoreEntry(BaseModel):
    """Score for one prompt within a ModelConfigResult."""

    prompt_id: str
    category: str
    scoring_mode: str
    score: PromptScore


class SummaryMetrics(BaseModel):
    """Aggregate metrics summarised across all prompts in a ModelConfigResult."""

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


class ModelConfigResult(BaseModel):
    """Scores for all prompts under one (model, config) pair."""

    model: str
    config_label: ConfigLabel
    n_runs: int
    prompt_scores: list[PromptScoreEntry]
    summary: SummaryMetrics


class RunDocument(BaseModel):
    """Top-level document written to disk. One file per invocation. Schema version 2.0."""

    schema_version: str = "2.0"
    run_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    results: list[ModelConfigResult]
