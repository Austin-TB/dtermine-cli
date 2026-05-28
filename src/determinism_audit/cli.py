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
from determinism_audit.config import (
    ConfigLabel,
    ResolvedModel,
    RunConfig,
    load_n_prompts_quick,
    resolve_models,
)
from determinism_audit.history import (
    RunDiff,
    append_history,
    compute_run_diff,
    find_prior_run,
    load_all_histories,
    load_prior_run,
)
from determinism_audit.report.html_writer import write_html_report
from determinism_audit.report.json_writer import build_model_config_result, write_run_document
from determinism_audit.result import ModelConfigResult, PromptResult, RunDocument, RunResult
from determinism_audit.runner import run_prompt

app = typer.Typer(
    name="determinism-audit",
    help="Measure LLM output determinism across providers and configurations.",
    add_completion=False,
)
console = Console(stderr=True)


def _sample_across_modes(prompts: list[Prompt], n_per_mode: int) -> list[Prompt]:
    """Take up to n_per_mode prompts from each scoring_mode bucket."""
    from collections import defaultdict

    buckets: dict[str, list[Prompt]] = defaultdict(list)
    for p in prompts:
        buckets[p.scoring_mode].append(p)
    result: list[Prompt] = []
    for bucket in buckets.values():
        result.extend(bucket[:n_per_mode])
    return result

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
    resolved: ResolvedModel,
    prompts: list[Prompt],
    config: RunConfig,
    n_runs: int,
    max_concurrency: int,
    progress: Progress,
    task_id: TaskID,
) -> list[PromptResult]:
    """Run all prompts for one (model, config) pair."""
    sem = asyncio.Semaphore(max_concurrency)
    model = resolved.model_id

    async def _bounded(prompt: Prompt) -> PromptResult:
        async with sem:
            runs: list[RunResult] = await run_prompt(
                model=model,
                prompt=prompt,
                config=config,
                n_runs=n_runs,
                supports_seed=resolved.supports_seed,
            )
            progress.advance(task_id)
            return PromptResult(
                prompt_id=prompt.id,
                category=prompt.category,
                scoring_mode=prompt.scoring_mode,
                runs=runs,
            )

    tasks = [asyncio.create_task(_bounded(p)) for p in prompts]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@app.command()
def main(
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            help="Comma-separated model IDs. Overrides models.config.json.",
        ),
    ] = None,
    models_config: Annotated[
        Path | None,
        typer.Option(
            "--models-config",
            help="JSON file listing models per provider (default: ./models.config.json if present).",
        ),
    ] = None,
    config: Annotated[
        str,
        typer.Option(
            "--config",
            help="Comma-separated config labels (A, B, C, D). Default: A,B,C,D",
        ),
    ] = "A,B,C,D",
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
        cli_models = _parse_models(models) if models else None
        resolved_models = resolve_models(
            cli_models=cli_models,
            models_config_path=models_config,
        )
    except (typer.BadParameter, ValueError, FileNotFoundError) as exc:
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
        n_per_mode = load_n_prompts_quick(models_config)
        all_prompts = _sample_across_modes(all_prompts, n_per_mode)
        n_runs = 3

    # --- Prepare output directory ---
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]determinism-audit[/] {__version__}  |  "
        f"{len(resolved_models)} model(s) x {len(run_configs)} config(s) x "
        f"{len(all_prompts)} prompts x {n_runs} runs"
    )

    mcr_list: list[ModelConfigResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        for resolved in resolved_models:
            for rc in run_configs:
                desc = f"{_slug(resolved.model_id)} / config-{rc.label.value}"
                task_id = progress.add_task(desc, total=len(all_prompts))

                prompt_results = asyncio.run(
                    _run_audit(
                        resolved=resolved,
                        prompts=all_prompts,
                        config=rc,
                        n_runs=n_runs,
                        max_concurrency=max_concurrency,
                        progress=progress,
                        task_id=task_id,
                    )
                )

                mcr = build_model_config_result(
                    prompt_results,
                    model=resolved.model_id,
                    config_label=rc.label.value,
                    n_runs=n_runs,
                )
                mcr_list.append(mcr)

    # --- Write single RunDocument ---
    ts = _timestamp()
    run_doc = RunDocument(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        results=mcr_list,
    )
    out_path = output_dir / f"{ts}-run.json"
    write_run_document(run_doc, out_path)
    console.print(f"  [green]✓[/] {out_path}")

    # --- Diff against prior run ---
    prior_path = find_prior_run(output_dir, exclude=out_path)
    prior_doc = load_prior_run(prior_path) if prior_path else None

    run_diffs: list[RunDiff] = []
    for mcr in mcr_list:
        prior_mcr = None
        if prior_doc is not None:
            from determinism_audit.history import _get_mcr

            prior_mcr = _get_mcr(prior_doc, mcr.model, mcr.config_label)

        run_diff = compute_run_diff(
            mcr,
            prior_mcr,
            current_path=out_path,
            prior_path=prior_path,
            current_timestamp=run_doc.timestamp,
            prior_timestamp=prior_doc.timestamp if prior_doc else None,
        )
        run_diffs.append(run_diff)
        append_history(
            output_dir / "history",
            _slug(mcr.model),
            mcr.model,
            run_diff,
        )

    # --- Generate HTML report ---
    histories = load_all_histories(output_dir / "history")
    html_path = output_dir / f"{ts}-report.html"
    write_html_report(run_doc, html_path, run_diffs=run_diffs, histories=histories)
    console.print(f"  [green]✓[/] {html_path}")

    console.print("[bold green]Done.[/]")
