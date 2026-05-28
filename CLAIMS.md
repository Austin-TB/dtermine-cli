# Pre-registered Claims

> All claims are pre-registered **before** any baseline data is collected (Phase 5).
> Status: **UNRESOLVED** until Phase 5/7 data is committed.

---

## C1 — Temperature-0 is not reliably byte-exact

**Claim:** Even at temperature=0 (Config A), fewer than 90 % of providers
achieve byte-exact rate (BER) ≥ 0.99 on the full 100-prompt canary suite.

**Rationale:** Sampling kernels, batching strategies, and floating-point
non-determinism mean that "temperature=0" is vendor-defined, not IEEE-defined.

**Verdict:** UNRESOLVED  
**Resolution target:** Phase 5

---

## C2 — Providers drift over calendar time

**Claim:** At least one provider will show a statistically significant drop
(p < 0.05, two-proportion z-test) in BER between the Phase 5 baseline and the
Phase 7 re-run (~45 days later).

**Rationale:** Silent model updates and infrastructure changes are common
practice; this claim quantifies whether they affect output stability.

**Verdict:** UNRESOLVED  
**Resolution target:** Phase 7

---

## C3 — Structured-output prompts are more stable than freeform

**Claim:** On Config A, the `structured_json` and `code` categories will have
BER ≥ 10 percentage points higher than `longform_summary` and
`ambiguous_open`.

**Rationale:** Constrained output spaces (valid JSON, compilable code) leave
the sampler less room to vary.

**Verdict:** UNRESOLVED  
**Resolution target:** Phase 5

---

## C4 — Seed parameter reduces variance measurably

**Claim:** Providers that honour an explicit `seed` parameter (Config D vs
Config B) will show BER ≥ 0.15 higher on matched prompts.

**Rationale:** Several providers (OpenAI, Mistral) expose a `seed` field;
this claim tests whether it actually works.

**Verdict:** UNRESOLVED  
**Resolution target:** Phase 5

---

## C5 — Open-source models hosted via Groq/Ollama are more byte-exact than frontier models

**Claim:** `groq/llama-3.1-8b-instant` and `ollama/llama3.2` will both
achieve BER ≥ 0.95 on Config A, outperforming at least two frontier models
from OpenAI or Anthropic.

**Rationale:** Smaller models with deterministic decoding and consistent
infrastructure may be more stable than large-scale, continuously-updated
frontier models.

**Verdict:** UNRESOLVED  
**Resolution target:** Phase 5

---

*Methodology and reproduction steps: see
[`plans/IMPLEMENTATION_PLAN.md`](plans/IMPLEMENTATION_PLAN.md).*
