# One image for both the API (default) and the Streamlit UI (compose overrides the command).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN pip install --no-cache-dir uv

# Dependencies first, from the lockfile, so code edits don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY streamlit_app.py pricing.toml ./
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev

# Writable state lives under /data so a volume can be mounted there.
# Secrets are NOT baked in: they arrive as env vars at run time.
ENV PERSIST_DIRECTORY=/data/chroma \
    EMBEDDING_CACHE_DIR=/data/embedding_cache
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=3)"

# Cloud Run injects PORT; locally it defaults to 8000.
CMD ["sh", "-c", "exec uvicorn financial_rag.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
