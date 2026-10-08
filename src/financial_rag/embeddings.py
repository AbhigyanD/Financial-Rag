"""Convert chunk text and queries into vector embeddings.

Why: one module owns the embedding model, so chunks and queries are always
embedded the same way, and the cache and batching live in one place.

Two providers, chosen by EMBEDDING_PROVIDER (see config.py):
- "local" (default): sentence-transformers/all-MiniLM-L6-v2 via fastembed,
  run on CPU with ONNX Runtime. Free, no API key, 384-dim vectors. The
  model (~90 MB) downloads once on first use.
- "openai": the OpenAI embeddings API (text-embedding-3-small, 1536 dims).

Query and chunk vectors MUST come from the same model, or similarity is
meaningless. storage.py names its collection after the model so vectors
from two models can never be mixed in one index.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from openai import OpenAI, OpenAIError

from financial_rag import embedding_cache
from financial_rag.config import settings
from financial_rag.loaders.chunker import Chunk
from financial_rag.observability import add_tokens, note

EMBEDDING_PROVIDER = settings.embedding_provider
EMBEDDING_MODEL = settings.embedding_model


class EmbeddedChunk(TypedDict):
    """A chunk plus its vector embedding, ready for storage."""

    page_number: int
    text: str
    chunk_index: int
    embedding: list[float]


class EmbeddingError(Exception):
    """Raised when embedding a chunk or query fails."""


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    if not settings.openai_api_key:
        raise EmbeddingError(
            "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Add it to .env, or use EMBEDDING_PROVIDER=local."
        )
    return OpenAI(timeout=settings.request_timeout_seconds)


@lru_cache(maxsize=1)
def _get_local_model():
    # Imported here so the openai provider never loads ONNX Runtime.
    from fastembed import TextEmbedding

    try:
        return TextEmbedding(EMBEDDING_MODEL, cache_dir=settings.model_cache_dir)
    except Exception as e:  # unknown model name, failed download, corrupt cache
        raise EmbeddingError(f"Could not load local embedding model '{EMBEDDING_MODEL}': {e}") from e


def _embed_uncached(texts: list[str]) -> list[list[float]]:
    """One provider call for a list of texts, vectors in input order."""
    if EMBEDDING_PROVIDER == "local":
        try:
            return [v.tolist() for v in _get_local_model().embed(texts)]
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Local embedding failed: {e}") from e

    if EMBEDDING_PROVIDER == "openai":
        try:
            response = _get_client().embeddings.create(input=texts, model=EMBEDDING_MODEL)
        except OpenAIError as e:
            raise EmbeddingError(f"OpenAI embedding call failed: {e}") from e
        add_tokens(embedding=response.usage.total_tokens)
        return [item.embedding for item in response.data]

    raise EmbeddingError(f"Unknown EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}'; use 'local' or 'openai'.")


def embed_text(text: str) -> list[float]:
    """Embed one string, from the on-disk cache when possible."""
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty or whitespace-only text.")

    cached = embedding_cache.get_many(EMBEDDING_MODEL, [text])
    if text in cached:
        note(query_embedding_cached=True)
        return cached[text]

    [vector] = _embed_uncached([text])
    embedding_cache.put_many(EMBEDDING_MODEL, {text: vector})
    return vector


def embed_chunks(chunks: list[Chunk], batch_size: int = 100) -> list[EmbeddedChunk]:
    """Embed chunks, returning them with their vectors attached.

    Cache first: chunk text embedded before (same model) costs nothing.
    Identical texts are embedded once. The rest go `batch_size` per call.
    """
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    vectors_by_text = embedding_cache.get_many(EMBEDDING_MODEL, texts)
    to_embed = sorted({t for t in texts if t not in vectors_by_text})
    note(chunks=len(texts), embedding_cache_hits=len(texts) - len(to_embed))

    for i in range(0, len(to_embed), batch_size):
        batch_texts = to_embed[i : i + batch_size]
        new_vectors = dict(zip(batch_texts, _embed_uncached(batch_texts)))
        embedding_cache.put_many(EMBEDDING_MODEL, new_vectors)
        vectors_by_text.update(new_vectors)

    return [
        EmbeddedChunk(
            page_number=c["page_number"],
            text=c["text"],
            chunk_index=c["chunk_index"],
            embedding=vectors_by_text[c["text"]],
        )
        for c in chunks
    ]


def embed_query(query: str) -> list[float]:
    """Embed a user's question. MiniLM and text-embedding-3-small are both
    symmetric (no separate query mode), so this is embed_text()."""
    return embed_text(query)
