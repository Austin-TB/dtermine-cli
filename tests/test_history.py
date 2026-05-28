"""Unit tests for history.py."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from determinism_audit.config import ConfigLabel
from determinism_audit.history import (
    EPSILON,
    ModelHistory,
    PromptDiff,
    RunDiff,
    _is_changed,
    _parse_ts,
    append_history,
    compute_run_diff,
    find_prior_run,
    load_all_histories,
    load_prior_run,
)
from determinism_audit.result import (
    ModelConfigResult,
    PromptScore,
    PromptScoreEntry,
    RunDocument,
    SummaryMetrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score(
    ber: float = 1.0,
    ssr: float | None = None,
    svr: float | None = None,
    div: float = 0.0,
) -> PromptScore:
    return PromptScore(
        byte_exact_rate=ber,
        byte_exact_ci_low=0.0,
        byte_exact_ci_high=1.0,
        semantic_stability_rate=ssr,
        semantic_stability_ci_low=0.0 if ssr is not None else None,
        semantic_stability_ci_high=1.0 if ssr is not None else None,
        structural_validity_rate=svr,
        structural_validity_ci_low=0.0 if svr is not None else None,
        structural_validity_ci_high=1.0 if svr is not None else None,
        divergence_index=div,
    )


def _entry(
    prompt_id: str = "p1",
    category: str = "factual",
    score: PromptScore | None = None,
) -> PromptScoreEntry:
    return PromptScoreEntry(
        prompt_id=prompt_id,
        category=category,
        scoring_mode="exact",
        score=score or _score(),
    )


def _summary() -> SummaryMetrics:
    return SummaryMetrics(
        byte_exact_rate=1.0,
        byte_exact_ci_low=0.0,
        byte_exact_ci_high=1.0,
        mean_divergence_index=0.0,
    )


def _mcr(
    entries: list[PromptScoreEntry],
    config_label: ConfigLabel = ConfigLabel.A,
    model: str = "groq/llama-3.1-8b-instant",
) -> ModelConfigResult:
    return ModelConfigResult(
        model=model,
        config_label=config_label,
        n_runs=5,
        prompt_scores=entries,
        summary=_summary(),
    )


def _run_doc(
    results: list[ModelConfigResult], timestamp: str = "2026-01-01T00:00:00Z"
) -> RunDocument:
    return RunDocument(run_id=str(uuid.uuid4()), timestamp=timestamp, results=results)


def _write_run_doc(path: Path, doc: RunDocument) -> None:
    path.write_text(doc.model_dump_json(), encoding="utf-8")


def _diff(
    current: ModelConfigResult,
    prior: ModelConfigResult | None,
    tmp_path: Path,
    current_ts: str = "2026-02-01T00:00:00Z",
    prior_ts: str | None = "2026-01-01T00:00:00Z",
) -> RunDiff:
    return compute_run_diff(
        current,
        prior,
        current_path=tmp_path / "cur.json",
        prior_path=tmp_path / "pri.json" if prior else None,
        current_timestamp=current_ts,
        prior_timestamp=prior_ts if prior else None,
    )


# ---------------------------------------------------------------------------
# _parse_ts
# ---------------------------------------------------------------------------


class TestParseTs:
    def test_valid_timestamp(self, tmp_path: Path) -> None:
        p = tmp_path / "20260101T120000Z-run.json"
        ts = _parse_ts(p)
        assert ts.year == 2026

    def test_no_timestamp_prefix(self, tmp_path: Path) -> None:
        p = tmp_path / "noprefix.json"
        ts = _parse_ts(p)
        import datetime

        assert ts == datetime.datetime.min.replace(tzinfo=datetime.UTC)


# ---------------------------------------------------------------------------
# _is_changed
# ---------------------------------------------------------------------------


class TestIsChanged:
    def test_all_zero_unchanged(self) -> None:
        assert not _is_changed(0.0, None, None, 0.0)

    def test_ber_delta_above_epsilon(self) -> None:
        assert _is_changed(EPSILON * 2, None, None, 0.0)

    def test_ber_delta_at_epsilon_unchanged(self) -> None:
        assert not _is_changed(EPSILON, None, None, 0.0)

    def test_divergence_delta_triggers(self) -> None:
        assert _is_changed(0.0, None, None, EPSILON * 2)

    def test_ssr_delta_triggers(self) -> None:
        assert _is_changed(0.0, EPSILON * 2, None, 0.0)

    def test_svr_delta_triggers(self) -> None:
        assert _is_changed(0.0, None, EPSILON * 2, 0.0)

    def test_none_ssr_ignored(self) -> None:
        assert not _is_changed(0.0, None, None, 0.0)


# ---------------------------------------------------------------------------
# compute_run_diff — first run (prior=None)
# ---------------------------------------------------------------------------


class TestComputeRunDiffFirstRun:
    def test_first_run_n_changed_zero(self, tmp_path: Path) -> None:
        current = _mcr([_entry("p1"), _entry("p2")])
        d = _diff(current, None, tmp_path)
        assert d.n_changed == 0
        assert d.prior_path is None
        assert d.prior_timestamp is None

    def test_first_run_all_added_status(self, tmp_path: Path) -> None:
        current = _mcr([_entry("p1"), _entry("p2")])
        d = _diff(current, None, tmp_path)
        assert all(pd.status == "added" for pd in d.prompt_diffs)
        assert all(not pd.changed for pd in d.prompt_diffs)

    def test_first_run_deltas_zero(self, tmp_path: Path) -> None:
        current = _mcr([_entry("p1", score=_score(ber=0.8, div=0.2))])
        d = _diff(current, None, tmp_path)
        assert d.prompt_diffs[0].ber_delta == 0.0
        assert d.prompt_diffs[0].divergence_delta == 0.0


# ---------------------------------------------------------------------------
# compute_run_diff — with prior
# ---------------------------------------------------------------------------


class TestComputeRunDiff:
    def test_no_change(self, tmp_path: Path) -> None:
        score = _score(ber=0.8, div=0.1)
        d = _diff(_mcr([_entry("p1", score=score)]), _mcr([_entry("p1", score=score)]), tmp_path)
        assert d.n_changed == 0
        assert d.prompt_diffs[0].status == "unchanged"

    def test_ber_changed(self, tmp_path: Path) -> None:
        d = _diff(
            _mcr([_entry("p1", score=_score(ber=0.9))]),
            _mcr([_entry("p1", score=_score(ber=0.7))]),
            tmp_path,
        )
        assert d.n_changed == 1
        assert d.prompt_diffs[0].status == "changed"
        assert d.prompt_diffs[0].ber_delta == pytest.approx(0.2)

    def test_prompt_added(self, tmp_path: Path) -> None:
        d = _diff(
            _mcr([_entry("p1"), _entry("p2")]),
            _mcr([_entry("p1")]),
            tmp_path,
        )
        added = [pd for pd in d.prompt_diffs if pd.prompt_id == "p2"]
        assert len(added) == 1
        assert added[0].status == "added"
        assert added[0].changed

    def test_prompt_removed(self, tmp_path: Path) -> None:
        d = _diff(
            _mcr([_entry("p1")]),
            _mcr([_entry("p1"), _entry("p2", score=_score(ber=0.5))]),
            tmp_path,
        )
        removed = [pd for pd in d.prompt_diffs if pd.prompt_id == "p2"]
        assert len(removed) == 1
        assert removed[0].status == "removed"
        assert removed[0].changed
        assert removed[0].ber_delta == pytest.approx(-0.5)

    def test_none_ssr_no_change(self, tmp_path: Path) -> None:
        score = _score(ber=1.0, ssr=None, div=0.0)
        d = _diff(_mcr([_entry("p1", score=score)]), _mcr([_entry("p1", score=score)]), tmp_path)
        assert d.n_changed == 0
        assert d.prompt_diffs[0].ssr_delta is None

    def test_ssr_delta_computed(self, tmp_path: Path) -> None:
        d = _diff(
            _mcr([_entry("p1", score=_score(ssr=0.9))]),
            _mcr([_entry("p1", score=_score(ssr=0.6))]),
            tmp_path,
        )
        assert d.prompt_diffs[0].ssr_delta == pytest.approx(0.3)
        assert d.n_changed == 1

    def test_epsilon_boundary_unchanged(self, tmp_path: Path) -> None:
        d = _diff(
            _mcr([_entry("p1", score=_score(ber=EPSILON))]),
            _mcr([_entry("p1", score=_score(ber=0.0))]),
            tmp_path,
        )
        assert d.n_changed == 0

    def test_epsilon_boundary_changed(self, tmp_path: Path) -> None:
        d = _diff(
            _mcr([_entry("p1", score=_score(ber=EPSILON * 2))]),
            _mcr([_entry("p1", score=_score(ber=0.0))]),
            tmp_path,
        )
        assert d.n_changed == 1

    def test_mixed_prompts(self, tmp_path: Path) -> None:
        d = _diff(
            _mcr([_entry("p1", score=_score(ber=1.0)), _entry("p2", score=_score(ber=0.5))]),
            _mcr([_entry("p1", score=_score(ber=1.0)), _entry("p2", score=_score(ber=0.9))]),
            tmp_path,
        )
        assert d.n_changed == 1
        p2 = next(pd for pd in d.prompt_diffs if pd.prompt_id == "p2")
        assert p2.ber_delta == pytest.approx(-0.4)

    def test_timestamps_set(self, tmp_path: Path) -> None:
        d = _diff(
            _mcr([]),
            _mcr([]),
            tmp_path,
            current_ts="2026-02-01T00:00:00Z",
            prior_ts="2026-01-01T00:00:00Z",
        )
        assert d.current_timestamp == "2026-02-01T00:00:00Z"
        assert d.prior_timestamp == "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# find_prior_run
# ---------------------------------------------------------------------------


class TestFindPriorRun:
    def _write_fake(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")

    def test_returns_none_when_no_files(self, tmp_path: Path) -> None:
        result = find_prior_run(tmp_path, tmp_path / "cur.json")
        assert result is None

    def test_returns_none_when_only_current(self, tmp_path: Path) -> None:
        cur = tmp_path / "20260201T000000Z-run.json"
        self._write_fake(cur)
        result = find_prior_run(tmp_path, cur)
        assert result is None

    def test_finds_single_prior(self, tmp_path: Path) -> None:
        cur = tmp_path / "20260201T000000Z-run.json"
        pri = tmp_path / "20260101T000000Z-run.json"
        self._write_fake(cur)
        self._write_fake(pri)
        result = find_prior_run(tmp_path, cur)
        assert result == pri

    def test_picks_newest_prior(self, tmp_path: Path) -> None:
        cur = tmp_path / "20260301T000000Z-run.json"
        p1 = tmp_path / "20260101T000000Z-run.json"
        p2 = tmp_path / "20260201T000000Z-run.json"
        for p in [cur, p1, p2]:
            self._write_fake(p)
        result = find_prior_run(tmp_path, cur)
        assert result == p2

    def test_ignores_non_run_json(self, tmp_path: Path) -> None:
        cur = tmp_path / "20260201T000000Z-run.json"
        other = tmp_path / "20260101T000000Z-report.html"
        self._write_fake(cur)
        self._write_fake(other)
        result = find_prior_run(tmp_path, cur)
        assert result is None


# ---------------------------------------------------------------------------
# load_prior_run
# ---------------------------------------------------------------------------


class TestLoadPriorRun:
    def test_valid_document(self, tmp_path: Path) -> None:
        doc = _run_doc([_mcr([_entry()])])
        path = tmp_path / "r.json"
        _write_run_doc(path, doc)
        loaded = load_prior_run(path)
        assert loaded is not None
        assert loaded.run_id == doc.run_id

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert load_prior_run(path) is None

    def test_schema_mismatch_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"schema_version": "0.9", "junk": true}', encoding="utf-8")
        assert load_prior_run(path) is None


# ---------------------------------------------------------------------------
# append_history + load_all_histories
# ---------------------------------------------------------------------------


def _make_run_diff(n_changed: int, prior_path: str | None = "prior.json") -> RunDiff:
    diffs = [
        PromptDiff(
            prompt_id=f"p{i}",
            category="factual",
            status="changed" if i < n_changed else "unchanged",
            ber_delta=0.1 if i < n_changed else 0.0,
            ssr_delta=None,
            svr_delta=None,
            divergence_delta=0.0,
            changed=i < n_changed,
        )
        for i in range(max(n_changed, 1))
    ]
    return RunDiff(
        model="groq/llama-3.1-8b-instant",
        config_label=ConfigLabel.A,
        current_path="current.json",
        prior_path=prior_path,
        current_timestamp="2026-02-01T00:00:00Z",
        prior_timestamp="2026-01-01T00:00:00Z" if prior_path else None,
        prompt_diffs=diffs,
        n_changed=n_changed,
    )


class TestAppendHistory:
    def test_no_change_returns_none(self, tmp_path: Path) -> None:
        diff = _make_run_diff(0)
        result = append_history(tmp_path / "history", "mymodel", "groq/llama", diff)
        assert result is None
        assert not (tmp_path / "history" / "mymodel.json").exists()

    def test_first_run_no_change_skipped(self, tmp_path: Path) -> None:
        diff = _make_run_diff(0, prior_path=None)
        result = append_history(tmp_path / "history", "mymodel", "groq/llama", diff)
        assert result is None

    def test_with_change_creates_file(self, tmp_path: Path) -> None:
        diff = _make_run_diff(1)
        result = append_history(tmp_path / "history", "mymodel", "groq/llama", diff)
        assert result is not None
        assert result.exists()

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        hdir = tmp_path / "history"
        diff = _make_run_diff(1)
        append_history(hdir, "mymodel", "groq/llama", diff)
        append_history(hdir, "mymodel", "groq/llama", diff)
        raw = json.loads((hdir / "mymodel.json").read_text())
        h = ModelHistory.model_validate(raw)
        assert len(h.entries) == 2

    def test_round_trip(self, tmp_path: Path) -> None:
        hdir = tmp_path / "history"
        diff = _make_run_diff(2)
        append_history(hdir, "mymodel", "groq/llama", diff)
        raw = json.loads((hdir / "mymodel.json").read_text())
        h = ModelHistory.model_validate(raw)
        assert h.model == "groq/llama"
        assert len(h.entries) == 1
        assert len(h.entries[0].changed_prompts) == 2

    def test_corrupt_history_reset(self, tmp_path: Path) -> None:
        hdir = tmp_path / "history"
        hdir.mkdir()
        (hdir / "mymodel.json").write_text("not json", encoding="utf-8")
        diff = _make_run_diff(1)
        result = append_history(hdir, "mymodel", "groq/llama", diff)
        assert result is not None
        raw = json.loads((hdir / "mymodel.json").read_text())
        h = ModelHistory.model_validate(raw)
        assert len(h.entries) == 1


class TestLoadAllHistories:
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        result = load_all_histories(tmp_path / "missing")
        assert result == {}

    def test_loads_multiple_models(self, tmp_path: Path) -> None:
        hdir = tmp_path / "history"
        for slug, model in [("modelA", "groq/a"), ("modelB", "groq/b")]:
            diff = _make_run_diff(1)
            append_history(hdir, slug, model, diff)
        result = load_all_histories(hdir)
        assert "groq/a" in result
        assert "groq/b" in result

    def test_skips_corrupt_files(self, tmp_path: Path) -> None:
        hdir = tmp_path / "history"
        hdir.mkdir()
        (hdir / "bad.json").write_text("bad", encoding="utf-8")
        result = load_all_histories(hdir)
        assert result == {}
