"""Pydantic schema for canary prompt entries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ScoringMode = Literal["exact", "structural", "semantic", "structural_semantic"]


class Prompt(BaseModel):
    """A single canary prompt entry loaded from prompts.jsonl."""

    id: str = Field(..., description="Unique stable identifier, e.g. 'factual-001'")
    category: str = Field(
        ...,
        description=(
            "One of: factual, structured_json, code, tool_calls, math, "
            "format_compliance, longform_summary, ambiguous_open"
        ),
    )
    prompt: str = Field(..., description="The prompt text sent to the model")
    scoring_mode: ScoringMode = Field(..., description="Which scorer to apply to the responses")
    max_tokens: int = Field(..., ge=1, le=4096)
    json_schema: dict[str, object] | None = Field(
        default=None,
        description="JSON Schema the model output must conform to (structured_json only)",
    )
