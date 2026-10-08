"""Central configuration, loaded from environment variables (and a local .env).

Import from this module instead of calling os.environ directly elsewhere,
so every tunable (model names, storage path, CORS origins) has one place
to look and one place to change.

Usage:
    from financial_rag.config import settings
    settings.openai_api_key
    settings.claude_model
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Loads variables from a .env file in the current working directory (if
# present) into os.environ. Real environment variables always win — this
# only fills in what isn't already set, so a shell `export` still works
# and CI/production environments that inject vars directly aren't affected.
load_dotenv()


def _env_list(name: str, default: list[str]) -> list[str]:
    """Read a comma-separated env var into a list, or fall back to default."""
    raw = os.environ.get(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- API keys (leave unset here; read fresh from the environment by
    # the SDKs themselves, e.g. OpenAI()/Anthropic() pick up
    # OPENAI_API_KEY/ANTHROPIC_API_KEY automatically). Exposed here too so
    # callers can check "is a key configured" without importing os. ---
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )

    # --- Stage 3: embeddings ---
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dimensions: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))
    )
    embedding_cache_dir: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_CACHE_DIR", "./embedding_cache")
    )

    # --- Stage 2: chunking ---
    chunk_max_size: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_MAX_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "100"))
    )

    # --- Stage 4: storage ---
    persist_directory: str = field(
        default_factory=lambda: os.environ.get("PERSIST_DIRECTORY", "./chroma_data")
    )
    collection_name: str = field(
        default_factory=lambda: os.environ.get("COLLECTION_NAME", "financial_documents")
    )

    # --- Stage 5: retrieval ---
    # Hybrid search (vector + BM25 merged by reciprocal rank fusion) and a
    # reranking pass are both off by default — pure vector search is the
    # tested, default-safe path; see retrieval.py for what each flag does.
    hybrid_retrieval: bool = field(
        default_factory=lambda: os.environ.get("HYBRID_RETRIEVAL", "false").lower() == "true"
    )
    rerank: bool = field(
        default_factory=lambda: os.environ.get("RERANK", "false").lower() == "true"
    )
    # Below this similarity, retrieve() treats the result as "no real match"
    # and generation abstains without calling the LLM — see llm.py::generate_answer.
    similarity_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SIMILARITY_THRESHOLD", "0.3"))
    )

    # --- Stage 6: LLM / prompt builder ---
    claude_model: str = field(
        default_factory=lambda: os.environ.get("CLAUDE_MODEL", "claude-opus-5")
    )
    claude_max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("CLAUDE_MAX_TOKENS", "1024"))
    )
    # Token budget for the CONTEXT portion of the prompt (excerpts), not the
    # whole request. Chunks are dropped (lowest-similarity first) until the
    # remaining context fits. See prompt_builder.py.
    context_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("CONTEXT_TOKEN_BUDGET", "4000"))
    )

    # --- Stage 7/8: API, logging, cost ---
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "60"))
    )

    # --- Stage 7: API ---
    # Comma-separated list, e.g. "http://localhost:3000,https://myapp.vercel.app".
    # Defaults to common local dev ports for a frontend.
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "CORS_ORIGINS", ["http://localhost:3000", "http://localhost:8501"]
        )
    )


settings = Settings()
