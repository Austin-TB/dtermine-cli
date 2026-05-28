"""Tests for config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from determinism_audit.config import (
    ConfigLabel,
    ProviderConfig,
    RunConfig,
    auto_detect_models,
    load_models_config,
    load_n_prompts,
    resolve_models,
)


def test_run_config_a() -> None:
    rc = RunConfig.from_label(ConfigLabel.A)
    assert rc.temperature == 0.0
    assert rc.seed == 42


def test_run_config_b() -> None:
    rc = RunConfig.from_label(ConfigLabel.B)
    assert rc.temperature == 1.0
    assert rc.seed is None


def test_run_config_c() -> None:
    rc = RunConfig.from_label(ConfigLabel.C)
    assert rc.temperature == 0.0
    assert rc.seed is None


def test_run_config_d() -> None:
    rc = RunConfig.from_label(ConfigLabel.D)
    assert rc.temperature == 1.0
    assert rc.seed == 42


def test_auto_detect_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    assert auto_detect_models() == []


def test_auto_detect_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for key in [
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    models = auto_detect_models()
    assert "openai/gpt-4o-mini" in models


def test_auto_detect_google_dedupes_gemini_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-2")
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "GROQ_API_KEY",
        "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    models = auto_detect_models()
    assert models.count("gemini/gemini-2.0-flash") == 1


def test_load_models_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "models.config.json"
    config_path.write_text(
        """
        {
          "providers": {
            "groq": {
              "env": "GROQ_API_KEY",
              "models": ["groq/llama-3.1-8b-instant", "groq/llama-3.3-70b-versatile"]
            },
            "google": {
              "env": "GOOGLE_API_KEY",
              "models": ["gemini/gemini-2.0-flash"]
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    models = load_models_config(config_path)
    assert [m.model_id for m in models] == [
        "groq/llama-3.1-8b-instant",
        "groq/llama-3.3-70b-versatile",
        "gemini/gemini-2.0-flash",
    ]
    gemini = next(m for m in models if m.provider == "google")
    assert gemini.supports_seed is False
    groq = next(m for m in models if m.provider == "groq")
    assert groq.supports_seed is True


def test_resolve_models_prefers_cli_over_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "models.config.json"
    config_path.write_text(
        '{"providers":{"groq":{"env":"GROQ_API_KEY","models":["groq/llama-3.1-8b-instant"]}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    resolved = resolve_models(
        cli_models=["gemini/gemini-2.0-flash"],
        models_config_path=config_path,
    )
    assert len(resolved) == 1
    assert resolved[0].model_id == "gemini/gemini-2.0-flash"
    assert resolved[0].supports_seed is False


def test_provider_config_from_env_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    pc = ProviderConfig.from_env(
        models=["openai/gpt-4o-mini"],
        config_labels=[ConfigLabel.A],
    )
    assert pc.models == ["openai/gpt-4o-mini"]
    assert len(pc.configs) == 1
    assert pc.configs[0].label == ConfigLabel.A


def test_resolve_models_requires_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="models.config.json"):
        resolve_models()


def test_load_models_config_short_model_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "models.config.json"
    config_path.write_text(
        """
        {
          "providers": {
            "groq": {
              "env": "GROQ_API_KEY",
              "models": ["llama-3.1-8b-instant"]
            },
            "google": {
              "env": "GOOGLE_API_KEY",
              "models": ["gemini-2.0-flash"]
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    models = load_models_config(config_path)
    assert models[0].model_id == "groq/llama-3.1-8b-instant"
    assert models[1].model_id == "gemini/gemini-2.0-flash"


def test_load_n_prompts(tmp_path: Path) -> None:
    config_path = tmp_path / "models.config.json"
    config_path.write_text(
        '{"n_prompts": 15, "providers": {"groq": {"env": "GROQ_API_KEY", "models": ["groq/llama-3.1-8b-instant"]}}}',
        encoding="utf-8",
    )
    assert load_n_prompts(config_path) == 15


def test_load_n_prompts_absent(tmp_path: Path) -> None:
    config_path = tmp_path / "models.config.json"
    config_path.write_text(
        '{"providers": {"groq": {"env": "GROQ_API_KEY", "models": ["groq/llama-3.1-8b-instant"]}}}',
        encoding="utf-8",
    )
    assert load_n_prompts(config_path) is None


def test_load_n_prompts_missing_file(tmp_path: Path) -> None:
    assert load_n_prompts(tmp_path / "nonexistent.json") is None


def test_provider_config_raises_no_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="models.config.json"):
        ProviderConfig.from_env()
