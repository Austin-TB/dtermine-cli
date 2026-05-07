"""Provider configuration and run-config definitions."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------


class ConfigLabel(StrEnum):
    """The four canonical sampling configurations."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class RunConfig(BaseModel):
    """Sampling parameters for a single audit run."""

    label: ConfigLabel
    temperature: float = Field(ge=0.0, le=2.0)
    seed: int | None = None

    @classmethod
    def from_label(cls, label: ConfigLabel) -> RunConfig:
        """Return the canonical RunConfig for a given label."""
        _CONFIGS: dict[ConfigLabel, RunConfig] = {
            ConfigLabel.A: cls(label=ConfigLabel.A, temperature=0.0, seed=42),
            ConfigLabel.B: cls(label=ConfigLabel.B, temperature=1.0, seed=None),
            ConfigLabel.C: cls(label=ConfigLabel.C, temperature=0.0, seed=None),
            ConfigLabel.D: cls(label=ConfigLabel.D, temperature=1.0, seed=42),
        }
        return _CONFIGS[label]


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_PROVIDER_MAP_PATH = Path(__file__).with_name("provider_map.json")

_OLLAMA_ENV = "OLLAMA_BASE_URL"
_OLLAMA_DEFAULT_MODEL = "ollama/llama3.2"


def _load_provider_map() -> dict[str, str]:
    """Load env var -> default model mapping from provider_map.json."""
    if not _PROVIDER_MAP_PATH.exists():
        return {}
    raw = _PROVIDER_MAP_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{_PROVIDER_MAP_PATH} must contain a JSON object")
    mapping: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"{_PROVIDER_MAP_PATH} contains invalid mapping entry")
        mapping[k] = v
    return mapping


def auto_detect_models() -> list[str]:
    """Return default model IDs for every provider key found in the environment."""
    models: list[str] = []
    provider_map = _load_provider_map()
    for env_var, model_id in provider_map.items():
        if os.environ.get(env_var, "").strip():
            models.append(model_id)
    if os.environ.get(_OLLAMA_ENV, "").strip():
        models.append(_OLLAMA_DEFAULT_MODEL)
    return models


class ProviderConfig(BaseModel):
    """Runtime-resolved provider configuration."""

    models: list[str]
    configs: list[RunConfig]

    @classmethod
    def from_env(
        cls,
        models: list[str] | None = None,
        config_labels: list[ConfigLabel] | None = None,
    ) -> ProviderConfig:
        """Build a ProviderConfig from environment, filling defaults as needed."""
        resolved_models = models if models else auto_detect_models()
        if not resolved_models:
            raise ValueError(
                "No provider API keys detected and no --models specified. "
                "Set at least one key in the environment or use --models."
            )
        resolved_labels = config_labels if config_labels else [ConfigLabel.A, ConfigLabel.B]
        resolved_configs = [RunConfig.from_label(lbl) for lbl in resolved_labels]
        return cls(models=resolved_models, configs=resolved_configs)
