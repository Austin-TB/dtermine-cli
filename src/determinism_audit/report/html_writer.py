"""Render an HTML report from a RunDocument and per-run diff data."""

from __future__ import annotations

import datetime
from datetime import UTC
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from determinism_audit.history import ModelHistory, RunDiff
from determinism_audit.result import ModelConfigResult, RunDocument, SummaryMetrics

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def _bar_cell(v: float | None) -> Markup:
    if v is None:
        return Markup('<span class="pill muted">n/a</span>')
    pct = max(0.0, min(1.0, v)) * 100
    cls = "good" if v >= 0.9 else ("warn" if v >= 0.5 else "bad")
    return Markup(
        f'<span class="bar"><span style="width: {pct:.1f}%"></span></span>'
        f'<span class="pill {cls}">{pct:.1f}%</span>'
    )


def _fmt_delta(v: float | None, *, invert: bool = False) -> Markup:
    if v is None:
        return Markup('<span class="delta-zero">—</span>')
    if abs(v) <= 1e-6:
        return Markup('<span class="delta-zero">±0.0 pp</span>')
    if invert:
        cls = "delta-neg" if v > 0 else "delta-pos"
    else:
        cls = "delta-pos" if v > 0 else "delta-neg"
    sign = "+" if v > 0 else ""
    return Markup(f'<span class="{cls}">{sign}{v * 100:.2f} pp</span>')


def _build_overview(results: list[ModelConfigResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        s: SummaryMetrics = r.summary
        rows.append(
            {
                "model": r.model,
                "config_label": r.config_label.value,
                "summary": s,
                "byte_exact_cell": _bar_cell(s.byte_exact_rate),
                "structural_cell": _bar_cell(s.structural_validity_rate),
                "semantic_cell": _bar_cell(s.semantic_stability_rate),
                "n_prompts": len(r.prompt_scores),
                "n_runs": r.n_runs,
            }
        )
    return rows


def _build_change_panels(run_diffs: list[RunDiff]) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    for rd in run_diffs:
        rows: list[dict[str, Any]] = []
        for d in rd.prompt_diffs:
            if not d.changed:
                continue
            rows.append(
                {
                    "prompt_id": d.prompt_id,
                    "category": d.category,
                    "status": d.status,
                    "ber_delta_cell": _fmt_delta(d.ber_delta),
                    "ssr_delta_cell": _fmt_delta(d.ssr_delta),
                    "svr_delta_cell": _fmt_delta(d.svr_delta),
                    "divergence_delta_cell": _fmt_delta(d.divergence_delta, invert=True),
                }
            )
        panels.append(
            {
                "model": rd.model,
                "config_label": rd.config_label.value,
                "prior_timestamp": rd.prior_timestamp,
                "current_timestamp": rd.current_timestamp,
                "n_changed": rd.n_changed,
                "n_total": len(rd.prompt_diffs),
                "rows": rows,
            }
        )
    return panels


def _build_history_panels(histories: dict[str, ModelHistory]) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    for model, history in sorted(histories.items()):
        entries: list[dict[str, Any]] = []
        for entry in reversed(history.entries):
            rows = [
                {
                    "prompt_id": d.prompt_id,
                    "category": d.category,
                    "status": d.status,
                    "ber_delta_cell": _fmt_delta(d.ber_delta),
                    "ssr_delta_cell": _fmt_delta(d.ssr_delta),
                    "svr_delta_cell": _fmt_delta(d.svr_delta),
                    "divergence_delta_cell": _fmt_delta(d.divergence_delta, invert=True),
                }
                for d in entry.changed_prompts
            ]
            entries.append(
                {
                    "timestamp": entry.timestamp,
                    "config_label": entry.config_label.value,
                    "n_changed": len(entry.changed_prompts),
                    "rows": rows,
                }
            )
        panels.append({"model": model, "entries": entries})
    return panels


def write_html_report(
    run_doc: RunDocument,
    output_path: Path,
    *,
    run_diffs: list[RunDiff] | None = None,
    histories: dict[str, ModelHistory] | None = None,
) -> Path:
    """Render an HTML report from *run_doc* and write to *output_path*."""
    if not run_doc.results:
        raise ValueError("RunDocument has no results to render")

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    models = sorted({r.model for r in run_doc.results})
    configs = sorted({r.config_label.value for r in run_doc.results})
    total_prompts = sum(len(r.prompt_scores) for r in run_doc.results)
    total_runs = sum(r.n_runs * len(r.prompt_scores) for r in run_doc.results)

    ctx: dict[str, Any] = {
        "generated_at": datetime.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "run_doc": run_doc,
        "models": models,
        "configs": configs,
        "total_prompts": total_prompts,
        "total_runs": total_runs,
        "overview": _build_overview(run_doc.results),
        "change_panels": _build_change_panels(run_diffs or []),
        "history_panels": _build_history_panels(histories or {}),
        "fmt_pct": _fmt_pct,
    }

    html = env.get_template(_TEMPLATE_NAME).render(**ctx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
