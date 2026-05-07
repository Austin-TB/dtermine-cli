"""Tests for config module."""

from __future__ import annotations

import os

import pytest

from determinism_audit.config import (
    ConfigLabel,
    ProviderConfig,
    RunConfig,
    auto_detect_models,
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
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY",
        "TOGETHER_API_KEY", "FIREWORKS_API_KEY", "GROQ_API_KEY", "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    assert auto_detect_models() == []


def test_auto_detect_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for key in [
        "ANTHROPIC_API_KEY", "MISTRAL_API_KEY",
        "TOGETHER_API_KEY", "FIREWORKS_API_KEY", "GROQ_API_KEY", "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    models = auto_detect_models()
    assert "openai/gpt-4o-mini" in models


def test_provider_config_from_env_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    pc = ProviderConfig.from_env(
        models=["openai/gpt-4o-mini"],
        config_labels=[ConfigLabel.A],
    )
    assert pc.models == ["openai/gpt-4o-mini"]
    assert len(pc.configs) == 1
    assert pc.configs[0].label == ConfigLabel.A


def test_provider_config_raises_no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY",
        "TOGETHER_API_KEY", "FIREWORKS_API_KEY", "GROQ_API_KEY", "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="No provider API keys detected"):
        ProviderConfig.from_env()
