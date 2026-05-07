"""Async litellm wrapper with retry policy and per-prompt timeout."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import litellm
import litellm.exceptions
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from determinism_audit.canary.schema import Prompt
from determinism_audit.config import RunConfig
from determinism_audit.result import RunResult

# ---------------------------------------------------------------------------
# Constants (locked Phase 1)
# ---------------------------------------------------------------------------

PROMPT_TIMEOUT_S = 60.0
_RETRY_ATTEMPTS = 4
_RETRY_WAIT_MIN = 1  # seconds
_RETRY_WAIT_MAX = 8  # seconds


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------


def _should_retry(exc: BaseException) -> bool:
    """Retry on rate-limit or 5xx; fail-fast on other 4xx."""
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return True
    if isinstance(exc, litellm.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status is not None:
            if status == 429:
                return True
            if status >= 500:
                return True
            # Other 4xx — fail fast
            return False
    # Network / timeout errors are retryable
    if isinstance(exc, litellm.exceptions.APIConnectionError | litellm.exceptions.Timeout):
        return True
    return False


def _build_error_payload(exc: BaseException) -> dict[str, Any]:
    code: int | None = getattr(exc, "status_code", None)
    return {
        "code": code,
        "message": str(exc),
        "type": type(exc).__name__,
    }


# ---------------------------------------------------------------------------
# Single-attempt call
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=_RETRY_WAIT_MIN, max=_RETRY_WAIT_MAX),
    reraise=True,
)
async def _call_once(
    model: str,
    prompt: Prompt,
    config: RunConfig,
) -> str:
    """Make one LLM API call and return the response text."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt.prompt}],
        "max_tokens": prompt.max_tokens,
        "temperature": config.temperature,
    }
    if config.seed is not None:
        kwargs["seed"] = config.seed

    response = await asyncio.wait_for(
        litellm.acompletion(**kwargs),
        timeout=PROMPT_TIMEOUT_S,
    )
    content: str = response.choices[0].message.content or ""
    return content


# ---------------------------------------------------------------------------
# Public interface (locked Phase 1)
# ---------------------------------------------------------------------------


async def run_prompt(
    model: str,
    prompt: Prompt,
    config: RunConfig,
    n_runs: int = 5,
) -> list[RunResult]:
    """
    Send *prompt* to *model* under *config* exactly *n_runs* times.

    Returns one :class:`RunResult` per run.  Errors are captured inside the
    result rather than raised, so a single failing run does not abort the batch.
    """
    results: list[RunResult] = []
    for run_index in range(n_runs):
        t0 = time.monotonic()
        response: str | None = None
        error: dict[str, Any] | None = None

        try:
            response = await _call_once(model=model, prompt=prompt, config=config)
        except Exception as exc:
            error = _build_error_payload(exc)

        latency_ms = (time.monotonic() - t0) * 1000.0

        results.append(
            RunResult(
                prompt_id=prompt.id,
                run_index=run_index,
                model=model,
                config_label=config.label,
                response=response,
                error=error,
                latency_ms=round(latency_ms, 2),
            )
        )
    return results
