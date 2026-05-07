# determinism-audit

> Measure how deterministic LLM outputs are across providers and configurations.

**Status:** Phase 1 — foundations & runner skeleton.

## Quick start

```bash
cp .env.example .env
# Edit .env — set at least one provider key

docker build -t determinism-audit:dev .
docker run --rm --env-file .env \
  -v $(pwd)/results:/app/results \
  determinism-audit:dev --models openai/gpt-4o-mini --quick
```

Results land in `results/<timestamp>-gpt-4o-mini.json`.

## What it measures

For each (model, configuration) pair the tool sends 25–100 "canary" prompts
**n** times and records whether the outputs are identical.  Metrics:

| Metric | Description |
|---|---|
| BER | Byte-exact rate — fraction of runs that produced identical bytes |
| SSR | Semantic-stability rate — cosine similarity ≥ 0.97 across runs |
| SVR | Structural-validity rate — valid JSON / parseable code |
| DI  | Divergence index — mean normalized Levenshtein of the worst pair |

## Configurations

| Label | Temperature | Seed |
|---|---|---|
| A | 0 | 42 |
| B | 1 | — |
| C | 0 | — |
| D | 1 | 42 |

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check src/
uv run mypy src/
```

See [`plans/IMPLEMENTATION_PLAN.md`](plans/IMPLEMENTATION_PLAN.md) for the
full 8-phase build plan and [`CLAIMS.md`](CLAIMS.md) for pre-registered
hypotheses.
