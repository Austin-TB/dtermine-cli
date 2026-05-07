"""Write and read AuditReport JSON documents.

This module is the single authoritative place that touches the disk format.
The schema it writes corresponds to schemas/result.schema.json (draft 2020-12).
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
    AuditReport,
    PromptResult,
    PromptScore,
    SummaryMetrics,
)

# Modes that require structural scoring
_STRUCTURAL_MODES: frozenset[ScoringMode] = frozenset({"structural", "structural_semantic"})
# Modes that require semantic scoring
_SEMANTIC_MODES: frozenset[ScoringMode] = frozenset({"semantic", "structural_semantic"})


def _score_prompt_result(pr: PromptResult) -> PromptScore:
    """Compute all applicable scores for a single PromptResult."""
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
    """Aggregate per-prompt scores into a single summary."""
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

    di_values = [pr.score.divergence_index for pr in prompt_results if pr.score is not None]
    mean_di = sum(di_values) / len(di_values) if di_values else 0.0

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


def score_and_write(report: AuditReport, output_path: Path) -> Path:
    """
    Score every PromptResult in *report*, attach the summary, then write the
    validated JSON document to *output_path*.

    Returns the resolved output path.
    """
    scored_results: list[PromptResult] = []
    for pr in report.prompt_results:
        score = _score_prompt_result(pr)
        scored_results.append(
            PromptResult(
                prompt_id=pr.prompt_id,
                category=pr.category,
                scoring_mode=pr.scoring_mode,
                runs=pr.runs,
                score=score,
            )
        )

    summary = _compute_summary(scored_results)

    final_report = AuditReport(
        schema_version=report.schema_version,
        run_id=report.run_id,
        timestamp=report.timestamp,
        model=report.model,
        config_label=report.config_label,
        n_runs=report.n_runs,
        prompt_results=scored_results,
        summary=summary,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_report.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def load_report(path: Path) -> AuditReport:
    """Load and validate an AuditReport from a JSON file on disk."""
    raw = path.read_text(encoding="utf-8")
    return AuditReport.model_validate(json.loads(raw))
