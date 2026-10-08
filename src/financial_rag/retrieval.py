"""Retrieve the most relevant chunks for a user's query.

Why: one place that turns a question into ranked chunks, so the prompt
builder and API never need to know which search method(s) ran.

Three modes, chosen by config flags (see config.py):
- Default: pure vector search (cosine similarity via Chroma).
- HYBRID_RETRIEVAL=true: vector search + BM25 keyword search, merged
  with reciprocal rank fusion (RRF). Helps on exact terms embeddings
  blur together — ticker symbols, account names, specific figures.
- RERANK=true: a re-ordering pass over the candidates. The current
  implementation is a NAIVE lexical-overlap heuristic, not a trained
  cross-encoder — see _lexical_rerank. Off by default.

Score semantics (important — they differ by mode):
- Vector-only: `similarity` is cosine similarity rescaled to [0, 1],
  higher = more relevant. The abstention threshold in llm.py applies to
  this number.
- Hybrid: `similarity` is replaced by the RRF fused score — a small
  positive number (max 2/61 ≈ 0.033 with k=60 and two rankings), only
  meaningful for ordering. The raw cosine similarity is preserved in
  `vector_similarity` so abstention still has a real signal to check.
"""

from __future__ import annotations

import re
from typing import TypedDict

from rank_bm25 import BM25Okapi

from financial_rag.config import settings
from financial_rag.embeddings import EmbeddingError, embed_query
from financial_rag.storage import StorageError, get_all_chunks, query_chunks

# Standard RRF constant from Cormack et al. (2009). Larger k flattens the
# advantage of a #1 rank; 60 is the commonly used default.
RRF_K = 60

# When hybrid or rerank is on, fetch more candidates than top_k from each
# source so fusion/reranking has something to choose between.
CANDIDATE_POOL = 20


class RetrievedChunk(TypedDict):
    """A chunk returned by retrieval, with relevance scores."""

    page_number: int
    text: str
    chunk_index: int
    source: str
    similarity: float  # ordering score — see module docstring for per-mode meaning
    vector_similarity: float  # cosine similarity in [0, 1]; 0.0 if only found by BM25


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


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25. Keeps digits, %, $, and . inside
    numbers so "$4.2B" and "12%" survive as tokens — figures are often
    exactly what a financial question is keyed on.
    """
    return re.findall(r"[a-z0-9$%][a-z0-9$%.,]*", text.lower())


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int]:
    return (chunk["source"], chunk["chunk_index"])


def _vector_search(query: str, fetch_k: int, source: str | None) -> list[RetrievedChunk]:
    try:
        query_embedding = embed_query(query)
    except EmbeddingError as e:
        raise RetrievalError(f"Failed to embed query: {e}") from e

    try:
        results = query_chunks(query_embedding, top_k=fetch_k, source=source)
    except StorageError as e:
        raise RetrievalError(f"Failed to query vector store: {e}") from e

    chunks = []
    for r in results:
        similarity = _distance_to_similarity(r["score"])
        chunks.append(
            RetrievedChunk(
                page_number=r["page_number"],
                text=r["text"],
                chunk_index=r["chunk_index"],
                source=r["source"],
                similarity=similarity,
                vector_similarity=similarity,
            )
        )
    return chunks


def _bm25_search(query: str, fetch_k: int, source: str | None) -> list[RetrievedChunk]:
    """Keyword search over every stored chunk (optionally one source).

    Rebuilds the BM25 index from a full collection scan on each call —
    simple and always consistent with the vector store, but O(corpus) per
    query. See storage.get_all_chunks for the scaling caveat.
    """
    try:
        corpus = get_all_chunks(source=source)
    except StorageError as e:
        raise RetrievalError(f"Failed to load corpus for keyword search: {e}") from e

    query_tokens = _tokenize(query)
    if not corpus or not query_tokens:
        return []

    bm25 = BM25Okapi([_tokenize(c["text"]) for c in corpus])
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)
    return [
        RetrievedChunk(
            page_number=c["page_number"],
            text=c["text"],
            chunk_index=c["chunk_index"],
            source=c["source"],
            similarity=float(score),  # raw BM25; only its rank is used downstream
            vector_similarity=0.0,  # filled in by fusion if vector search also found it
        )
        for c, score in ranked[:fetch_k]
        if score > 0  # zero score = no query term appears at all; not a match
    ]


def _fuse_rankings(
    rankings: list[list[RetrievedChunk]], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Merge several ranked lists with reciprocal rank fusion.

    Each chunk's fused score is the sum over every list it appears in of
    1 / (k + rank). Uses rank, not raw score, because cosine similarity
    and BM25 live on incomparable scales — RRF sidesteps normalizing
    them. A chunk ranked well by both methods beats one ranked #1 by
    only one.
    """
    fused_scores: dict[tuple[str, int], float] = {}
    best_vector_sim: dict[tuple[str, int], float] = {}
    by_key: dict[tuple[str, int], RetrievedChunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            key = _chunk_key(chunk)
            by_key.setdefault(key, chunk)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            best_vector_sim[key] = max(best_vector_sim.get(key, 0.0), chunk["vector_similarity"])

    ordered = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
    return [
        RetrievedChunk(
            **{
                **by_key[key],
                "similarity": fused_scores[key],
                "vector_similarity": best_vector_sim[key],
            }
        )
        for key in ordered
    ]


def _lexical_rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """NAIVE placeholder reranker — not a trained model.

    Re-orders candidates by how many distinct query tokens appear in each
    chunk, breaking ties with the existing ordering score. A production
    system would call a cross-encoder here (e.g. a bge-reranker model or
    a hosted rerank API), which reads query and chunk together and is
    far more accurate. This exists so the RERANK flag and the call site
    are real and testable; it should not be described as a reranker in
    the ML sense.
    """
    query_tokens = set(_tokenize(query))

    def overlap(chunk: RetrievedChunk) -> int:
        return len(query_tokens & set(_tokenize(chunk["text"])))

    return sorted(chunks, key=lambda c: (overlap(c), c["similarity"]), reverse=True)


def retrieve(
    query: str, top_k: int = 5, source: str | None = None
) -> list[RetrievedChunk]:
    """Find the top_k chunks most relevant to a natural-language query.

    Args:
        query: The user's question/search text.
        top_k: Number of chunks to return.
        source: If given, restrict results to chunks from this document only.

    Returns:
        RetrievedChunk dicts ordered best-first. See the module docstring
        for what `similarity` means under each config mode.

    Raises:
        RetrievalError: if embedding, vector search, or keyword search fails.
    """
    use_pool = settings.hybrid_retrieval or settings.rerank
    fetch_k = max(top_k, CANDIDATE_POOL) if use_pool else top_k

    results = _vector_search(query, fetch_k, source)

    if settings.hybrid_retrieval:
        keyword_results = _bm25_search(query, fetch_k, source)
        results = _fuse_rankings([results, keyword_results])

    if settings.rerank:
        results = _lexical_rerank(query, results)

    return results[:top_k]
