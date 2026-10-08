"""On-disk cache for embeddings, keyed by a hash of (model, text).

Why: re-embedding unchanged chunks on every ingest wastes real OpenAI API
calls and money. This is a minimal sqlite-backed cache — stdlib only, no
new dependency — that embeddings.py checks before calling the API, and
fills in after. Keying on the model name too means switching embedding
models doesn't return stale, incompatible vectors from a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing

from financial_rag.config import settings


def _cache_path() -> str:
    os.makedirs(settings.embedding_cache_dir, exist_ok=True)
    return os.path.join(settings.embedding_cache_dir, "embeddings.sqlite3")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_cache_path())
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embedding_cache ("
        "key TEXT PRIMARY KEY, embedding TEXT NOT NULL)"
    )
    return conn


def _key(model: str, text: str) -> str:
    """Hash of (model, text). Same text under a different model is a
    cache miss on purpose — vectors from different models aren't
    comparable, so reusing one would silently corrupt similarity scores.
    """
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def get_many(model: str, texts: list[str]) -> dict[str, list[float]]:
    """Return {text: embedding} for every text already cached under `model`.

    Texts not found in the cache are simply absent from the returned dict
    (not an error) — the caller treats a missing key as a cache miss.
    """
    if not texts:
        return {}
    keys = {_key(model, t): t for t in texts}
    with closing(_connect()) as conn:
        placeholders = ",".join("?" * len(keys))
        rows = conn.execute(
            f"SELECT key, embedding FROM embedding_cache WHERE key IN ({placeholders})",
            list(keys.keys()),
        ).fetchall()
    return {keys[key]: json.loads(embedding) for key, embedding in rows}


def put_many(model: str, items: dict[str, list[float]]) -> None:
    """Store {text: embedding} pairs under `model`. Overwrites on conflict
    (harmless — a given (model, text) pair always maps to the same vector).
    """
    if not items:
        return
    with closing(_connect()) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (key, embedding) VALUES (?, ?)",
            [(_key(model, text), json.dumps(vector)) for text, vector in items.items()],
        )
        conn.commit()
