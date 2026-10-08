"""Convert chunk text into vector embeddings for semantic search.

Why: one module owns the embedding model, so chunks and queries are always
embedded the same way, and the cache and batching live in one place.

This is stage 3 of the ingest pipeline: given a list[Chunk] from stage 2
(chunker), produce a vector embedding for each chunk's text. Embeddings are
what make retrieval possible — at query time we embed the user's question
the same way and compare vectors for similarity.

Design notes:
- Embedding a query MUST use the exact same model/dimensions as embedding
  chunks, or similarity scores are meaningless.
- Batch requests where the provider supports it — embedding one chunk at a
  time is slow and wastes API calls.
- Keep the embedding vector as a plain list[float] so it's JSON-serializable
  and storage-agnostic (stage 4 can pickle it, put it in Chroma, etc.).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from openai import OpenAI, OpenAIError

from financial_rag import embedding_cache
from financial_rag.config import settings
from financial_rag.loaders.chunker import Chunk
from financial_rag.observability import add_tokens, note


class EmbeddedChunk(TypedDict):
    """A chunk plus its vector embedding, ready for storage."""

    page_number: int
    text: str
    chunk_index: int
    embedding: list[float]


class EmbeddingError(Exception):
    """Raised when embedding a chunk or query fails."""


# Model is fixed at import time: mixing vectors from two different models
# (or dimensions) in the same index makes similarity scores meaningless.
# Configurable via EMBEDDING_MODEL / EMBEDDING_DIMENSIONS in .env — see config.py.
EMBEDDING_MODEL = settings.embedding_model
EMBEDDING_DIMENSIONS = settings.embedding_dimensions


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Create/return the shared OpenAI client, cached for the process.

    Reads OPENAI_API_KEY from the environment (the OpenAI SDK does this
    automatically). Raises EmbeddingError with a clear message if the key
    is missing rather than letting the client fail on first use.
    """
    if not settings.openai_api_key:
        raise EmbeddingError(
            "OPENAI_API_KEY is not set. Add it to a .env file or export it "
            "in your environment before calling embed_text/embed_chunks."
        )
    return OpenAI(timeout=settings.request_timeout_seconds)


def embed_text(text: str) -> list[float]:
    """Embed a single string of text into a vector.

    Checks the on-disk cache (embedding_cache.py) first — a repeated
    query or an unchanged chunk doesn't cost an API call.

    Raises:
        EmbeddingError: if `text` is empty/whitespace-only, or the API
            call fails (auth, rate limit, timeout, etc.).
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty or whitespace-only text.")

    cached = embedding_cache.get_many(EMBEDDING_MODEL, [text])
    if text in cached:
        note(query_embedding_cached=True)
        return cached[text]

    client = _get_client()
    try:
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    except OpenAIError as e:
        raise EmbeddingError(f"OpenAI embedding call failed: {e}") from e

    add_tokens(embedding=response.usage.total_tokens)
    vector = response.data[0].embedding
    embedding_cache.put_many(EMBEDDING_MODEL, {text: vector})
    return vector


def embed_chunks(
    chunks: list[Chunk], batch_size: int = 100
) -> list[EmbeddedChunk]:
    """Embed a list of chunks, returning them with their vectors attached.

    Two cost-saving steps before any API call is made:
    1. Cache lookup — chunks whose exact text was embedded before (under
       the same model) are served from embedding_cache.py, no API call.
    2. Real batching — remaining chunks are sent `batch_size` at a time
       in ONE `client.embeddings.create(input=[...])` call per batch
       (OpenAI's embeddings endpoint accepts a list and returns vectors
       in the same order), not one call per chunk.

    Args:
        chunks: List of Chunk dicts from chunker.py.
        batch_size: Number of (cache-miss) chunks to send per API call.

    Returns:
        A list of EmbeddedChunk dicts (same fields as Chunk, plus `embedding`).

    Raises:
        EmbeddingError: if a batch API call fails.
    """
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    vectors_by_text = embedding_cache.get_many(EMBEDDING_MODEL, texts)

    # De-duplicate: if the same text appears in multiple chunks (or more
    # than once in `texts`), only embed it once.
    to_embed = sorted({t for t in texts if t not in vectors_by_text})
    note(chunks=len(texts), embedding_cache_hits=len(texts) - len(to_embed))
    if not to_embed:
        return _attach(chunks, vectors_by_text)

    client = _get_client()
    for i in range(0, len(to_embed), batch_size):
        batch_texts = to_embed[i : i + batch_size]
        try:
            response = client.embeddings.create(input=batch_texts, model=EMBEDDING_MODEL)
        except OpenAIError as e:
            raise EmbeddingError(
                f"OpenAI batch embedding call failed on batch {i // batch_size}: {e}"
            ) from e

        add_tokens(embedding=response.usage.total_tokens)
        new_vectors = {
            text: item.embedding for text, item in zip(batch_texts, response.data)
        }
        embedding_cache.put_many(EMBEDDING_MODEL, new_vectors)
        vectors_by_text.update(new_vectors)

    return _attach(chunks, vectors_by_text)


def _attach(chunks: list[Chunk], vectors_by_text: dict[str, list[float]]) -> list[EmbeddedChunk]:
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
    """Embed a user's query string for similarity comparison against chunks.

    IMPORTANT: Must use the same model as embed_text/embed_chunks, or
    similarity scores will be meaningless.

    text-embedding-3-small is symmetric (no separate query/document mode),
    so this is a thin wrapper around embed_text(). If the model were ever
    swapped for an asymmetric one, the query-specific mode would go here.
    """
    return embed_text(query)
   

def get_embedding_dimensions() -> int:
    """Return the dimensionality of vectors produced by this module.

    Used by stage 4 (vector storage) to validate/initialize the DB
    schema/index without embedding a throwaway string.
    """
    return EMBEDDING_DIMENSIONS
