"""Typer CLI entry point for determinism-audit."""

from __future__ import annotations

import asyncio
import datetime
import re
import uuid
from datetime import UTC
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from determinism_audit import __version__
from determinism_audit.canary.loader import load_prompts
from determinism_audit.canary.schema import Prompt
from determinism_audit.config import ConfigLabel, RunConfig, auto_detect_models
from determinism_audit.report.json_writer import score_and_write
from determinism_audit.result import AuditReport, PromptResult, RunResult
from determinism_audit.runner import run_prompt

app = typer.Typer(
    name="determinism-audit",
    help="Measure LLM output determinism across providers and configurations.",
    add_completion=False,
)
console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_MODEL_RE = re.compile(r"^[a-zA-Z0-9_\-/.]+$")


def _parse_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.split(",") if m.strip()]
    for m in models:
        if not _VALID_MODEL_RE.match(m):
            raise typer.BadParameter(f"Invalid model identifier: {m!r}")
    return models


def _parse_configs(raw: str) -> list[ConfigLabel]:
    labels: list[ConfigLabel] = []
    for part in raw.split(","):
        part = part.strip().upper()
        try:
            labels.append(ConfigLabel(part))
        except ValueError as err:
            raise typer.BadParameter(
                f"Unknown config {part!r}. Valid options: A, B, C, D."
            ) from err
    return labels


def _slug(model: str) -> str:
    """Turn 'openai/gpt-4o-mini' into 'openai-gpt-4o-mini'."""
    return model.replace("/", "-").replace(":", "-")


def _timestamp() -> str:
    return datetime.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------


async def _run_audit(
    model: str,
    prompts: list[Prompt],
    config: RunConfig,
    n_runs: int,
    max_concurrency: int,
    progress: Progress,
    task_id: TaskID,
) -> list[PromptResult]:
    """Run all prompts for one (model, config) pair."""
    sem = asyncio.Semaphore(max_concurrency)
    prompt_results: list[PromptResult] = []

    async def _bounded(prompt: Prompt) -> PromptResult:
        async with sem:
            runs: list[RunResult] = await run_prompt(
                model=model, prompt=prompt, config=config, n_runs=n_runs
            )
            progress.advance(task_id)
            return PromptResult(
                prompt_id=prompt.id,
                category=prompt.category,
                scoring_mode=prompt.scoring_mode,
                runs=runs,
            )

    tasks = [asyncio.create_task(_bounded(p)) for p in prompts]
    prompt_results = list(await asyncio.gather(*tasks))
    return prompt_results


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@app.command()
def main(
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            help="Comma-separated model IDs. Defaults to auto-detect from env.",
        ),
    ] = None,
    config: Annotated[
        str,
        typer.Option(
            "--config",
            help="Comma-separated config labels (A, B, C, D). Default: A,B",
        ),
    ] = "A,B",
    quick: Annotated[
        bool,
        typer.Option("--quick", help="20 prompts x 3 runs instead of full suite."),
    ] = False,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory to write result JSON files."),
    ] = Path("results"),
    max_concurrency: Annotated[
        int,
        typer.Option("--max-concurrency", help="Max parallel requests per provider.", min=1),
    ] = 4,
    version: Annotated[
        bool,
        typer.Option("--version", help="Print version and exit."),
    ] = False,
) -> None:
    """Run the LLM determinism audit and write result JSON(s) to OUTPUT_DIR."""

    if version:
        typer.echo(f"determinism-audit {__version__}")
        raise typer.Exit()

    # --- Resolve models ---
    try:
        if models:
            resolved_models = _parse_models(models)
        else:
            resolved_models = auto_detect_models()
            if not resolved_models:
                console.print(
                    "[bold red]Error:[/] No provider keys found in environment "
                    "and --models was not specified."
                )
                raise typer.Exit(1)
    except typer.BadParameter as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    # --- Resolve configs ---
    try:
        config_labels = _parse_configs(config)
    except typer.BadParameter as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    run_configs = [RunConfig.from_label(lbl) for lbl in config_labels]

    # --- Load prompts ---
    all_prompts = load_prompts()
    n_runs = 5
    if quick:
        all_prompts = all_prompts[:20]
        n_runs = 3

    # --- Prepare output directory ---
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]determinism-audit[/] {__version__}  |  "
        f"{len(resolved_models)} model(s) x {len(run_configs)} config(s) x "
        f"{len(all_prompts)} prompts x {n_runs} runs"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        for model_id in resolved_models:
            for rc in run_configs:
                desc = f"{_slug(model_id)} / config-{rc.label.value}"
                task_id = progress.add_task(desc, total=len(all_prompts))

                prompt_results = asyncio.run(
                    _run_audit(
                        model=model_id,
                        prompts=all_prompts,
                        config=rc,
                        n_runs=n_runs,
                        max_concurrency=max_concurrency,
                        progress=progress,
                        task_id=task_id,
                    )
                )

                report = AuditReport(
                    run_id=str(uuid.uuid4()),
                    model=model_id,
                    config_label=rc.label,
                    n_runs=n_runs,
                    prompt_results=prompt_results,
                )

                ts = _timestamp()
                filename = f"{ts}-{_slug(model_id)}-config{rc.label.value}.json"
                out_path = output_dir / filename

                score_and_write(report, out_path)
                console.print(f"  [green]✓[/] {out_path}")

    console.print("[bold green]Done.[/]")
