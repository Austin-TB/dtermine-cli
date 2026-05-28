FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install system deps needed by sentence-transformers / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install runtime dependencies (no dev extras)
RUN uv pip install --system --no-cache \
    typer \
    "litellm>=1.40.0,<2.0.0" \
    "pydantic>=2.7.0,<3.0.0" \
    tenacity \
    rich \
    jinja2 \
    "sentence-transformers>=3.0.0"

# Pre-download the embedding model so the container is offline-capable at score time
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/opt/models")
EOF

# Copy source
COPY src/ ./src/

# Install the package itself (no build isolation needed, editable-ish via pip)
RUN uv pip install --system --no-cache -e .

# Results volume
VOLUME ["/app/results"]

ENTRYPOINT ["determinism-audit"]
