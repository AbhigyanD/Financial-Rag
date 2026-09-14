"""Retrieve the most relevant chunks for a user's query.

This is stage 5 of the pipeline: given a natural-language query, embed it
the same way chunks were embedded (stage 3) and search the vector store
(stage 4) for the closest matches. This module is the glue between the
two — callers (stage 6's prompt builder, or an API layer) should only
ever need to call `retrieve()`.

Design notes:
- Chroma's query_chunks() returns a raw *distance* (lower = more similar).
  That's an implementation detail of the storage backend and shouldn't
  leak to callers — this module converts it into a similarity score in
  [0, 1] where HIGHER = more relevant, which is what most people expect
  and what a prompt-builder/UI will want to reason about or threshold on.
- storage.py configures the Chroma collection to use cosine distance
  (metadata={"hnsw:space": "cosine"}), which lands in [0, 2]. That's what
  _distance_to_similarity() assumes.
"""

from __future__ import annotations

from typing import TypedDict

from financial_rag.embeddings import EmbeddingError, embed_query
from financial_rag.storage import StorageError, query_chunks


class RetrievedChunk(TypedDict):
    """A chunk returned by retrieval, with a normalized similarity score."""

    page_number: int
    text: str
    chunk_index: int
    source: str
    similarity: float  # in [0, 1], HIGHER = more relevant


class RetrievalError(Exception):
    """Raised when retrieval fails."""


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma's cosine distance (range [0, 2]) into [0, 1] similarity.

    cosine_distance = 1 - cosine_similarity, so cosine_similarity is
    naturally in [-1, 1]; rescaling by (2 - distance) / 2 maps it to
    [0, 1] with 1.0 = identical vectors, 0.0 = maximally dissimilar.
    """
    similarity = (2 - distance) / 2
    # Clamp defensively: floating point noise could nudge results a hair
    # outside [0, 1], and a differently-configured collection (e.g. an
    # older one still on L2) could produce distances outside [0, 2].
    return max(0.0, min(1.0, similarity))


def retrieve(
    query: str, top_k: int = 5, source: str | None = None
) -> list[RetrievedChunk]:
    """Find the top_k chunks most relevant to a natural-language query.

    Args:
        query: The user's question/search text.
        top_k: Number of chunks to return.
        source: If given, restrict results to chunks from this document only.

    Returns:
        A list of RetrievedChunk dicts ordered by relevance (best match
        first), each with a `similarity` score in [0, 1] (higher = better).

    Raises:
        RetrievalError: if embedding the query or querying the store fails.
    """
    try:
        query_embedding = embed_query(query)
    except EmbeddingError as e:
        raise RetrievalError(f"Failed to embed query: {e}") from e

    try:
        results = query_chunks(query_embedding, top_k=top_k, source=source)
    except StorageError as e:
        raise RetrievalError(f"Failed to query vector store: {e}") from e

    return [
        RetrievedChunk(
            page_number=r["page_number"],
            text=r["text"],
            chunk_index=r["chunk_index"],
            source=r["source"],
            similarity=_distance_to_similarity(r["score"]),
        )
        for r in results
    ]
