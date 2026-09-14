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

    # --- Stage 4: storage ---
    persist_directory: str = field(
        default_factory=lambda: os.environ.get("PERSIST_DIRECTORY", "./chroma_data")
    )
    collection_name: str = field(
        default_factory=lambda: os.environ.get("COLLECTION_NAME", "financial_documents")
    )

    # --- Stage 6: LLM ---
    claude_model: str = field(
        default_factory=lambda: os.environ.get("CLAUDE_MODEL", "claude-opus-5")
    )
    claude_max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("CLAUDE_MAX_TOKENS", "1024"))
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
