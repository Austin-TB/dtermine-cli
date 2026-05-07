"""Semantic scorer using sentence-transformers (all-MiniLM-L6-v2).

The model is loaded once per process from /opt/models if available,
falling back to the HuggingFace Hub for local development.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_BAKED_MODEL_PATH = Path("/opt/models") / _MODEL_NAME

DEFAULT_THRESHOLD = 0.97


@lru_cache(maxsize=1)
def _load_model() -> "SentenceTransformer":
    """Load the embedding model, preferring the baked-in Docker path."""
    from sentence_transformers import SentenceTransformer

    model_path = str(_BAKED_MODEL_PATH) if _BAKED_MODEL_PATH.exists() else _MODEL_NAME
    return SentenceTransformer(model_path)  # type: ignore[no-any-return]


def _cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def encode(texts: list[str]) -> list[list[float]]:
    """Encode *texts* into embedding vectors."""
    model = _load_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [list(map(float, e)) for e in embeddings]


def score_semantic(a: str, b: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Return True iff cosine similarity between *a* and *b* >= *threshold*."""
    embs = encode([a, b])
    similarity = _cosine(embs[0], embs[1])
    return similarity >= threshold


def pairwise_similarities(responses: list[str]) -> list[float]:
    """
    Compute all pairwise cosine similarities for *responses*.

    Returns a flat list of similarity scores for every unique (i, j) pair
    where i < j.
    """
    if len(responses) < 2:
        return []
    embeddings = encode(responses)
    scores: list[float] = []
    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(_cosine(embeddings[i], embeddings[j]))
    return scores


def all_above_threshold(responses: list[str], threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Return True iff all pairwise similarities are >= *threshold*."""
    sims = pairwise_similarities(responses)
    if not sims:
        return True
    return all(s >= threshold for s in sims)
