"""Convert chunk text into vector embeddings for semantic search.

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

from financial_rag.loaders.chunker import Chunk


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
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Create/return the shared OpenAI client, cached for the process.

    Reads OPENAI_API_KEY from the environment (the OpenAI SDK does this
    automatically). Raises EmbeddingError with a clear message if the key
    is missing rather than letting the client fail on first use.
    """
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        raise EmbeddingError(
            "OPENAI_API_KEY is not set. Export it in your environment "
            "before calling embed_text/embed_chunks."
        )
    return OpenAI()


def embed_text(text: str) -> list[float]:
    """Embed a single string of text into a vector.

    Raises:
        EmbeddingError: if `text` is empty/whitespace-only, or the API
            call fails (auth, rate limit, timeout, etc.).
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty or whitespace-only text.")

    client = _get_client()
    try:
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    except OpenAIError as e:
        raise EmbeddingError(f"OpenAI embedding call failed: {e}") from e

    return response.data[0].embedding


def embed_chunks(
    chunks: list[Chunk], batch_size: int = 100
) -> list[EmbeddedChunk]:
    """Embed a list of chunks, returning them with their vectors attached.

    Args:
        chunks: List of Chunk dicts from chunker.py.
        batch_size: Number of chunks to send per API call (if the provider
            supports batching). Ignored for local models if not applicable.

    Returns:
        A list of EmbeddedChunk dicts (same fields as Chunk, plus `embedding`).

    TODO:
    - Split `chunks` into batches of `batch_size`.
    - For each batch, extract the `text` field and call the provider's
      batch embedding endpoint (falls back to embed_text() in a loop if
      the provider has no batch API).
    - Zip the returned vectors back onto the original chunks in order —
      double check the provider preserves input order.
    - Consider a retry/backoff wrapper for rate limits.
    - Consider a progress callback/logging for large documents (many chunks).
    - Raise EmbeddingError on failure; decide whether a partial failure
      should abort the whole batch or return what succeeded.
    """

    if not chunks:  # if the list is empty, return an empty list of embeddings
        return []

    embedded_chunks: list[EmbeddedChunk] = []
    for i in range(0, len(chunks), batch_size):  # finds the start of each batch
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = [embed_text(text) for text in texts]

        embedded_chunks.extend(
            {
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"],
                "embedding": embedding,
            }
            for chunk, embedding in zip(batch, embeddings)
        )

        progress = (i + len(batch)) / len(chunks) * 100
        print(f"Progress: {progress:.2f}% ({i + len(batch)}/{len(chunks)})")

    return embedded_chunks


def embed_query(query: str) -> list[float]:
    """Embed a user's query string for similarity comparison against chunks.

    IMPORTANT: Must use the same model as embed_text/embed_chunks, or
    similarity scores will be meaningless.

    TODO:
    - Thin wrapper around embed_text(query) — some providers distinguish
      between "document" and "query" embedding modes (e.g. asymmetric
      embedding models); if so, use the query-specific mode here.
    """
    return embed_text(query)

    


def get_embedding_dimensions() -> int:
    """Return the dimensionality of vectors produced by this module.

    TODO:
    - Return the constant for the chosen model (e.g. EMBEDDING_DIMENSIONS).
    - Useful for stage 4 (vector storage) to validate/initialize the DB
      schema/index without embedding a throwaway string.
    """
    return EMBEDDING_DIMENSIONS
