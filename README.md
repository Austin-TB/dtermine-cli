# determinism-audit

> Measure how deterministic LLM outputs are across providers and configurations.

**Status:** Phase 1 — foundations & runner skeleton.

## Quick start

```bash
cp .env.example .env
# Edit .env — set at least one provider key

cp models.config.example.json models.config.json
# Edit models.config.json — list which models to run per provider

docker build -t determinism-audit:dev .
docker run --rm --env-file .env \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models.config.json:/app/models.config.json \
  determinism-audit:dev --quick
```

`models.config.json` is required. Every model listed under a provider whose API key
is set in `.env` is audited (multiple models per provider). Model ids may be short
names (`llama-3.1-8b-instant`) or full LiteLLM ids (`groq/llama-3.1-8b-instant`).
Gemini models skip `seed` on configs A/D automatically.

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
