"""Write and read RunDocument JSON files (schema version 2.0).

This module is the single authoritative place that touches the disk format.
"""

from __future__ import annotations

import json
from pathlib import Path

from determinism_audit.canary.schema import ScoringMode
from determinism_audit.metrics import (
    byte_exact_rate,
    divergence_index,
    semantic_stability_rate,
    structural_validity_rate,
)
from determinism_audit.result import (
    ModelConfigResult,
    PromptResult,
    PromptScore,
    PromptScoreEntry,
    RunDocument,
    SummaryMetrics,
)

_STRUCTURAL_MODES: frozenset[ScoringMode] = frozenset({"structural", "structural_semantic"})
_SEMANTIC_MODES: frozenset[ScoringMode] = frozenset({"semantic", "structural_semantic"})


def _score_prompt_result(pr: PromptResult) -> PromptScore:
    runs = pr.runs
    mode: ScoringMode = pr.scoring_mode  # type: ignore[assignment]

    ber, ber_lo, ber_hi = byte_exact_rate(runs)
    di = divergence_index(runs)

    svr: float | None = None
    svr_lo: float | None = None
    svr_hi: float | None = None
    if mode in _STRUCTURAL_MODES:
        svr, svr_lo, svr_hi = structural_validity_rate(runs)

    ssr: float | None = None
    ssr_lo: float | None = None
    ssr_hi: float | None = None
    if mode in _SEMANTIC_MODES:
        ssr, ssr_lo, ssr_hi = semantic_stability_rate(runs)

    return PromptScore(
        byte_exact_rate=round(ber, 6),
        byte_exact_ci_low=round(ber_lo, 6),
        byte_exact_ci_high=round(ber_hi, 6),
        structural_validity_rate=round(svr, 6) if svr is not None else None,
        structural_validity_ci_low=round(svr_lo, 6) if svr_lo is not None else None,
        structural_validity_ci_high=round(svr_hi, 6) if svr_hi is not None else None,
        semantic_stability_rate=round(ssr, 6) if ssr is not None else None,
        semantic_stability_ci_low=round(ssr_lo, 6) if ssr_lo is not None else None,
        semantic_stability_ci_high=round(ssr_hi, 6) if ssr_hi is not None else None,
        divergence_index=round(di, 6),
    )


def _compute_summary(prompt_results: list[PromptResult]) -> SummaryMetrics:
    all_runs = [r for pr in prompt_results for r in pr.runs]

    ber, ber_lo, ber_hi = byte_exact_rate(all_runs)

    structural_runs = [
        r for pr in prompt_results if pr.scoring_mode in _STRUCTURAL_MODES for r in pr.runs
    ]
    svr: float | None = None
    svr_lo: float | None = None
    svr_hi: float | None = None
    if structural_runs:
        svr, svr_lo, svr_hi = structural_validity_rate(structural_runs)

    semantic_runs = [
        r for pr in prompt_results if pr.scoring_mode in _SEMANTIC_MODES for r in pr.runs
    ]
    ssr: float | None = None
    ssr_lo: float | None = None
    ssr_hi: float | None = None
    if semantic_runs:
        ssr, ssr_lo, ssr_hi = semantic_stability_rate(semantic_runs)

    scores = [_score_prompt_result(pr) for pr in prompt_results]
    mean_di = sum(s.divergence_index for s in scores) / len(scores) if scores else 0.0

    return SummaryMetrics(
        byte_exact_rate=round(ber, 6),
        byte_exact_ci_low=round(ber_lo, 6),
        byte_exact_ci_high=round(ber_hi, 6),
        structural_validity_rate=round(svr, 6) if svr is not None else None,
        structural_validity_ci_low=round(svr_lo, 6) if svr_lo is not None else None,
        structural_validity_ci_high=round(svr_hi, 6) if svr_hi is not None else None,
        semantic_stability_rate=round(ssr, 6) if ssr is not None else None,
        semantic_stability_ci_low=round(ssr_lo, 6) if ssr_lo is not None else None,
        semantic_stability_ci_high=round(ssr_hi, 6) if ssr_hi is not None else None,
        mean_divergence_index=round(mean_di, 6),
    )


def build_model_config_result(
    prompt_results: list[PromptResult],
    model: str,
    config_label: str,
    n_runs: int,
) -> ModelConfigResult:
    """Score all PromptResults and produce a ModelConfigResult."""
    from determinism_audit.config import ConfigLabel

    prompt_scores: list[PromptScoreEntry] = []
    for pr in prompt_results:
        score = _score_prompt_result(pr)
        prompt_scores.append(
            PromptScoreEntry(
                prompt_id=pr.prompt_id,
                category=pr.category,
                scoring_mode=pr.scoring_mode,
                score=score,
            )
        )

    summary = _compute_summary(prompt_results)

    return ModelConfigResult(
        model=model,
        config_label=ConfigLabel(config_label),
        n_runs=n_runs,
        prompt_scores=prompt_scores,
        summary=summary,
    )


def write_run_document(doc: RunDocument, output_path: Path) -> Path:
    """Write a validated RunDocument to disk as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def load_run_document(path: Path) -> RunDocument:
    """Load and validate a RunDocument from a JSON file on disk."""
    raw = path.read_text(encoding="utf-8")
    return RunDocument.model_validate(json.loads(raw))
