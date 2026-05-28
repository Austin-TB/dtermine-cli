"""Per-run diff computation and per-model change history persistence."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from determinism_audit.config import ConfigLabel
from determinism_audit.result import ModelConfigResult, PromptScore, RunDocument

logger = logging.getLogger(__name__)

EPSILON = 1e-6

_TS_RE = re.compile(r"^(\d{8}T\d{6}Z)-")


def _parse_ts(path: Path) -> datetime:
    m = _TS_RE.match(path.name)
    if not m:
        return datetime.min.replace(tzinfo=UTC)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


class PromptDiff(BaseModel):
    prompt_id: str
    category: str
    status: Literal["changed", "added", "removed", "unchanged"]
    ber_delta: float
    ssr_delta: float | None
    svr_delta: float | None
    divergence_delta: float
    changed: bool


class RunDiff(BaseModel):
    model: str
    config_label: ConfigLabel
    current_path: str
    prior_path: str | None
    current_timestamp: str
    prior_timestamp: str | None
    prompt_diffs: list[PromptDiff]
    n_changed: int


class RunChangeEntry(BaseModel):
    timestamp: str
    config_label: ConfigLabel
    run_file: str
    prior_file: str | None
    changed_prompts: list[PromptDiff]


class ModelHistory(BaseModel):
    schema_version: str = "1.0"
    model: str
    entries: list[RunChangeEntry]


def find_prior_run(results_dir: Path, exclude: Path) -> Path | None:
    """Return the most recent *-run.json in results_dir, excluding the current file."""
    candidates = [
        p
        for p in results_dir.glob("*-run.json")
        if p.resolve() != exclude.resolve() and _TS_RE.match(p.name)
    ]
    if not candidates:
        return None
    return max(candidates, key=_parse_ts)


def load_prior_run(path: Path) -> RunDocument | None:
    from determinism_audit.report.json_writer import load_run_document

    try:
        return load_run_document(path)
    except Exception as exc:
        logger.warning("Skipping prior run document %s: %s", path, exc)
        return None


def _get_mcr(doc: RunDocument, model: str, config_label: ConfigLabel) -> ModelConfigResult | None:
    for mcr in doc.results:
        if mcr.model == model and mcr.config_label == config_label:
            return mcr
    return None


def _float_delta(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None:
        return None
    return current - prior


def _is_changed(
    ber_delta: float,
    ssr_delta: float | None,
    svr_delta: float | None,
    divergence_delta: float,
) -> bool:
    if abs(ber_delta) > EPSILON or abs(divergence_delta) > EPSILON:
        return True
    if ssr_delta is not None and abs(ssr_delta) > EPSILON:
        return True
    if svr_delta is not None and abs(svr_delta) > EPSILON:
        return True
    return False


def _score_val(score: PromptScore | None, field: str) -> float | None:
    if score is None:
        return None
    return getattr(score, field, None)


def compute_run_diff(
    current: ModelConfigResult,
    prior: ModelConfigResult | None,
    current_path: Path,
    prior_path: Path | None,
    current_timestamp: str,
    prior_timestamp: str | None,
) -> RunDiff:
    if prior is None:
        diffs = [
            PromptDiff(
                prompt_id=e.prompt_id,
                category=e.category,
                status="added",
                ber_delta=0.0,
                ssr_delta=None,
                svr_delta=None,
                divergence_delta=0.0,
                changed=False,
            )
            for e in current.prompt_scores
        ]
        return RunDiff(
            model=current.model,
            config_label=current.config_label,
            current_path=str(current_path),
            prior_path=None,
            current_timestamp=current_timestamp,
            prior_timestamp=None,
            prompt_diffs=diffs,
            n_changed=0,
        )

    current_by_id = {e.prompt_id: e for e in current.prompt_scores}
    prior_by_id = {e.prompt_id: e for e in prior.prompt_scores}
    all_ids = sorted(set(current_by_id) | set(prior_by_id))

    diffs = []
    for pid in all_ids:
        c_entry = current_by_id.get(pid)
        p_entry = prior_by_id.get(pid)

        if c_entry is None:
            p_score = p_entry.score  # type: ignore[union-attr]
            diffs.append(
                PromptDiff(
                    prompt_id=pid,
                    category=p_entry.category,  # type: ignore[union-attr]
                    status="removed",
                    ber_delta=-(_score_val(p_score, "byte_exact_rate") or 0.0),
                    ssr_delta=(
                        -v
                        if (v := _score_val(p_score, "semantic_stability_rate")) is not None
                        else None
                    ),
                    svr_delta=(
                        -v
                        if (v := _score_val(p_score, "structural_validity_rate")) is not None
                        else None
                    ),
                    divergence_delta=-(_score_val(p_score, "divergence_index") or 0.0),
                    changed=True,
                )
            )
            continue

        if p_entry is None:
            c_score = c_entry.score
            diffs.append(
                PromptDiff(
                    prompt_id=pid,
                    category=c_entry.category,
                    status="added",
                    ber_delta=_score_val(c_score, "byte_exact_rate") or 0.0,
                    ssr_delta=_score_val(c_score, "semantic_stability_rate"),
                    svr_delta=_score_val(c_score, "structural_validity_rate"),
                    divergence_delta=_score_val(c_score, "divergence_index") or 0.0,
                    changed=True,
                )
            )
            continue

        c_score = c_entry.score
        p_score = p_entry.score

        ber_delta = (
            _float_delta(
                _score_val(c_score, "byte_exact_rate"),
                _score_val(p_score, "byte_exact_rate"),
            )
            or 0.0
        )
        ssr_delta = _float_delta(
            _score_val(c_score, "semantic_stability_rate"),
            _score_val(p_score, "semantic_stability_rate"),
        )
        svr_delta = _float_delta(
            _score_val(c_score, "structural_validity_rate"),
            _score_val(p_score, "structural_validity_rate"),
        )
        divergence_delta = (
            _float_delta(
                _score_val(c_score, "divergence_index"),
                _score_val(p_score, "divergence_index"),
            )
            or 0.0
        )

        changed = _is_changed(ber_delta, ssr_delta, svr_delta, divergence_delta)
        status: Literal["changed", "added", "removed", "unchanged"] = (
            "changed" if changed else "unchanged"
        )

        diffs.append(
            PromptDiff(
                prompt_id=pid,
                category=c_entry.category,
                status=status,
                ber_delta=ber_delta,
                ssr_delta=ssr_delta,
                svr_delta=svr_delta,
                divergence_delta=divergence_delta,
                changed=changed,
            )
        )

    n_changed = sum(1 for d in diffs if d.changed)
    return RunDiff(
        model=current.model,
        config_label=current.config_label,
        current_path=str(current_path),
        prior_path=str(prior_path) if prior_path else None,
        current_timestamp=current_timestamp,
        prior_timestamp=prior_timestamp,
        prompt_diffs=diffs,
        n_changed=n_changed,
    )


def append_history(
    history_dir: Path,
    model_slug: str,
    model: str,
    run_diff: RunDiff,
) -> Path | None:
    if run_diff.n_changed == 0:
        return None

    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{model_slug}.json"

    if history_path.exists():
        try:
            raw = json.loads(history_path.read_text(encoding="utf-8"))
            history = ModelHistory.model_validate(raw)
        except Exception as exc:
            logger.warning("History file corrupt, resetting %s: %s", history_path, exc)
            history = ModelHistory(model=model, entries=[])
    else:
        history = ModelHistory(model=model, entries=[])

    changed_prompts = [d for d in run_diff.prompt_diffs if d.changed]
    entry = RunChangeEntry(
        timestamp=run_diff.current_timestamp,
        config_label=run_diff.config_label,
        run_file=run_diff.current_path,
        prior_file=run_diff.prior_path,
        changed_prompts=changed_prompts,
    )
    history.entries.append(entry)

    tmp_fd, tmp_str = tempfile.mkstemp(dir=history_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(history.model_dump_json(indent=2))
        os.replace(tmp_str, history_path)
    except Exception:
        try:
            os.unlink(tmp_str)
        except OSError:
            pass
        raise

    return history_path


def load_all_histories(history_dir: Path) -> dict[str, ModelHistory]:
    if not history_dir.exists():
        return {}
    result: dict[str, ModelHistory] = {}
    for p in history_dir.glob("*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            h = ModelHistory.model_validate(raw)
            result[h.model] = h
        except Exception as exc:
            logger.warning("Skipping history file %s: %s", p, exc)
    return result
