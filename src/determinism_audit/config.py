"""Provider configuration and run-config definitions."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# Map alternate env var names to the canonical key LiteLLM reads first.
_ENV_ALIASES: dict[str, list[str]] = {
    "GOOGLE_API_KEY": ["GEMINI_API_KEY"],
    "GEMINI_API_KEY": ["GOOGLE_API_KEY"],
}

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


def _env_is_set(env_var: str) -> bool:
    """True if *env_var* or any configured alias is set and non-empty."""
    if os.environ.get(env_var, "").strip():
        return True
    for alias in _ENV_ALIASES.get(env_var, []):
        if os.environ.get(alias, "").strip():
            return True
    return False


def auto_detect_models() -> list[str]:
    """Return default model IDs for every provider key found in the environment."""
    models: list[str] = []
    seen: set[str] = set()
    provider_map = _load_provider_map()
    for env_var, model_id in provider_map.items():
        if _env_is_set(env_var) and model_id not in seen:
            models.append(model_id)
            seen.add(model_id)
    if os.environ.get(_OLLAMA_ENV, "").strip() and _OLLAMA_DEFAULT_MODEL not in seen:
        models.append(_OLLAMA_DEFAULT_MODEL)
    return models


# ---------------------------------------------------------------------------
# models.config.json — multiple models per provider
# ---------------------------------------------------------------------------

DEFAULT_MODELS_CONFIG_PATH = Path("models.config.json")

# LiteLLM provider prefix when the model id in config has no "prefix/name" form.
_DEFAULT_LITELLM_PREFIX: dict[str, str] = {
    "google": "gemini",
}

# Providers that reject the seed parameter (config A / D).
_NO_SEED_PREFIXES: frozenset[str] = frozenset({"gemini"})


class ResolvedModel(BaseModel):
    """A model entry ready for the audit runner."""

    provider: str
    model_id: str
    supports_seed: bool = True


class ProviderModels(BaseModel):
    """Models to audit for one provider when its API key is present."""

    env: str = Field(..., description="Environment variable holding the provider API key")
    models: list[str] = Field(..., min_length=1)
    prefix: str | None = Field(
        default=None,
        description="LiteLLM provider prefix when model ids omit it (default: provider key).",
    )
    supports_seed: bool | None = Field(
        default=None,
        description="Whether seed is sent for configs A/D (default: false for gemini).",
    )

    @field_validator("models")
    @classmethod
    def _non_empty_models(cls, models: list[str]) -> list[str]:
        cleaned = [m.strip() for m in models if m.strip()]
        if not cleaned:
            raise ValueError("models must contain at least one non-empty model id")
        return cleaned


class ModelsConfigFile(BaseModel):
    """Top-level schema for models.config.json."""

    providers: dict[str, ProviderModels]
    n_prompts: int | None = Field(default=None, ge=1)

    def resolve_models(self) -> list[ResolvedModel]:
        """Return all models whose provider env key (or alias) is set."""
        resolved: list[ResolvedModel] = []
        for provider_key, provider in self.providers.items():
            if not _env_is_set(provider.env):
                continue
            litellm_prefix = provider.prefix or _DEFAULT_LITELLM_PREFIX.get(
                provider_key, provider_key
            )
            supports_seed = _supports_seed(provider_key, litellm_prefix, provider.supports_seed)
            for raw in provider.models:
                resolved.append(
                    ResolvedModel(
                        provider=provider_key,
                        model_id=_normalize_model_id(raw, litellm_prefix),
                        supports_seed=supports_seed,
                    )
                )
        return resolved


def _normalize_model_id(raw: str, litellm_prefix: str) -> str:
    """Turn config model ids into LiteLLM model strings (prefix/name)."""
    if "/" in raw:
        return raw
    return f"{litellm_prefix}/{raw}"


def _supports_seed(
    provider_key: str,
    litellm_prefix: str,
    explicit: bool | None,
) -> bool:
    if explicit is not None:
        return explicit
    if provider_key == "google" or litellm_prefix in _NO_SEED_PREFIXES:
        return False
    return True


def _resolved_from_cli_model(model: str) -> ResolvedModel:
    prefix = model.split("/", 1)[0] if "/" in model else "unknown"
    supports_seed = prefix not in _NO_SEED_PREFIXES
    return ResolvedModel(provider=prefix, model_id=model, supports_seed=supports_seed)


def load_models_config(path: Path) -> list[ResolvedModel]:
    """Load and resolve models from a models.config.json file."""
    if not path.exists():
        raise FileNotFoundError(f"Models config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = ModelsConfigFile.model_validate(raw)
    models = config.resolve_models()
    if not models:
        raise ValueError(
            f"No models resolved from {path}: set at least one provider API key "
            "matching a provider entry, or add models under a provider whose key is set."
        )
    return models


def load_n_prompts(path: Path | None = None) -> int | None:
    """Return n_prompts from the models config file, or None if unset/missing."""
    config_path = path or DEFAULT_MODELS_CONFIG_PATH
    if not config_path.exists():
        return None
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = ModelsConfigFile.model_validate(raw)
    return config.n_prompts


def resolve_models(
    *,
    cli_models: list[str] | None = None,
    models_config_path: Path | None = None,
) -> list[ResolvedModel]:
    """
    Resolve the model list for an audit run.

    Priority: explicit CLI ``--models`` > ``models.config.json`` (required by default).
    """
    if cli_models:
        return [_resolved_from_cli_model(m) for m in cli_models]
    config_path = models_config_path or DEFAULT_MODELS_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Models config not found: {config_path}. "
            "Copy models.config.example.json to models.config.json, "
            "or pass --models-config / --models."
        )
    return load_models_config(config_path)


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
        resolved = resolve_models(cli_models=models)
        resolved_models = [m.model_id for m in resolved]
        resolved_labels = config_labels if config_labels else [ConfigLabel.A, ConfigLabel.B]
        resolved_configs = [RunConfig.from_label(lbl) for lbl in resolved_labels]
        return cls(models=resolved_models, configs=resolved_configs)
