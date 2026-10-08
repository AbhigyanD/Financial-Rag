"""Central configuration, loaded from environment variables (and a local .env).

Why: every tunable has one place to look and one place to change, and
secrets only ever come from the environment, never from code.

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


_DEFAULT_EMBEDDING_MODELS = {
    "local": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
}


def _default_embedding_model() -> str:
    provider = os.environ.get("EMBEDDING_PROVIDER", "local").lower()
    return os.environ.get("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODELS.get(provider, ""))


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
    groq_api_key: str | None = field(default_factory=lambda: os.environ.get("GROQ_API_KEY"))

    # --- Stage 3: embeddings ---
    # "local" runs a small model on CPU via fastembed (free, no key);
    # "openai" calls the OpenAI embeddings API (needs OPENAI_API_KEY).
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_PROVIDER", "local").lower()
    )
    embedding_model: str = field(default_factory=lambda: _default_embedding_model())
    # Where fastembed keeps downloaded model files (local provider only).
    model_cache_dir: str = field(
        default_factory=lambda: os.environ.get("MODEL_CACHE_DIR", "./model_cache")
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
    # If no retrieved chunk reaches this similarity, generation abstains
    # without calling the LLM (llm.retrieval_is_weak). It only saves a
    # call: gate 2 (the model's own "not found" plus citation checks) still
    # runs for anything that passes. Similarity is (1 + cosine) / 2, with
    # local MiniLM. On the eval corpus: answerable 0.697-0.878, off-topic
    # 0.506-0.560. In real use, one-word questions about an uploaded resume
    # scored 0.570-0.605 and were wrongly blocked at the earlier 0.62.
    # 0.58 is still above every off-topic score measured.
    similarity_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SIMILARITY_THRESHOLD", "0.58"))
    )

    # --- Stage 6: LLM / prompt builder ---
    # "groq" (default; OpenAI-compatible API, has a free tier) or "anthropic".
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "groq").lower()
    )
    # openai/gpt-oss-120b: tied with qwen/qwen3.8-27b on the eval (15/15
    # correct, 5/5 abstentions, 2/2 injections), but with no pause between
    # questions it never failed on Groq's free tier while qwen hit 429s on
    # 7/20. See EVAL_RESULTS.md. llama-3.3-70b-versatile wasn't available.
    groq_model: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    claude_model: str = field(
        default_factory=lambda: os.environ.get("CLAUDE_MODEL", "claude-opus-5")
    )
    # Output cap for either provider. On claude-opus-5 thinking is on by
    # default and counts against it, so a small cap can cut an answer off.
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("LLM_MAX_TOKENS", "4096"))
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
    # Per-token prices for the logged cost ESTIMATE (see observability.py).
    pricing_file: str = field(
        default_factory=lambda: os.environ.get("PRICING_FILE", "pricing.toml")
    )
    # Uploads above this are rejected with 413 before any parsing or embedding.
    max_upload_mb: float = field(
        default_factory=lambda: float(os.environ.get("MAX_UPLOAD_MB", "25"))
    )

    # --- Stage 7: API ---
    # Comma-separated list, e.g. "http://localhost:3000,https://myapp.vercel.app".
    # Defaults to common local dev ports for a frontend.
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "CORS_ORIGINS", ["http://localhost:3000", "http://localhost:8501"]
        )
    )

    @property
    def llm_model(self) -> str:
        """The model actually used for generation, given llm_provider."""
        return self.claude_model if self.llm_provider == "anthropic" else self.groq_model


settings = Settings()
