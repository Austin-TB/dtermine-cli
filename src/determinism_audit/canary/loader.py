"""Load and validate canary prompts from prompts.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from determinism_audit.canary.schema import Prompt

_DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompts.jsonl"


def load_prompts(path: Path = _DEFAULT_PROMPTS_PATH) -> list[Prompt]:
    """
    Read *path* (newline-delimited JSON), validate each entry against the
    :class:`Prompt` schema, and return the list.

    Raises :exc:`ValueError` if any line fails validation.
    """
    prompts: list[Prompt] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
            try:
                prompts.append(Prompt.model_validate(data))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: schema validation failed — {exc}") from exc
    return prompts
