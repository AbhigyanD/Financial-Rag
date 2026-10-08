"""Tests for retrieval.py — similarity math, BM25, RRF fusion, reranking,
and the config-flag routing in retrieve(). No vector store or API calls:
storage and embedding functions are patched with fakes.
"""

import dataclasses

import pytest

import financial_rag.retrieval as R
from financial_rag.retrieval import (
    RetrievedChunk,
    _distance_to_similarity,
    _fuse_rankings,
    _lexical_rerank,
    _tokenize,
)


def _chunk(source="r.pdf", idx=0, text="x", similarity=0.5, vector_similarity=None, page=1):
    return RetrievedChunk(
        page_number=page,
        text=text,
        chunk_index=idx,
        source=source,
        similarity=similarity,
        vector_similarity=similarity if vector_similarity is None else vector_similarity,
    )


@pytest.fixture
def flags(monkeypatch):
    """Set retrieval config flags for one test. Settings is frozen, so
    swap the module's reference for a modified copy instead of mutating.
    """

    def _set(**overrides):
        monkeypatch.setattr(R, "settings", dataclasses.replace(R.settings, **overrides))

    return _set


# --- distance -> similarity -------------------------------------------------


def test_zero_distance_is_perfect_similarity():
    assert _distance_to_similarity(0.0) == 1.0


def test_max_cosine_distance_is_zero_similarity():
    assert _distance_to_similarity(2.0) == 0.0


def test_mid_distance_is_mid_similarity():
    assert _distance_to_similarity(1.0) == 0.5


def test_similarity_decreases_as_distance_increases():
    assert _distance_to_similarity(0.2) > _distance_to_similarity(0.8)


def test_out_of_range_distance_is_clamped():
    assert _distance_to_similarity(-0.5) == 1.0
    assert _distance_to_similarity(3.0) == 0.0


# --- tokenizer --------------------------------------------------------------


def test_tokenize_keeps_financial_figures_intact():
    tokens = _tokenize("Revenue was $4.2B, up 12% YoY")
    assert "$4.2b," in tokens or "$4.2b" in tokens
    assert "12%" in tokens


def test_tokenize_lowercases():
    assert _tokenize("EBITDA") == ["ebitda"]


# --- reciprocal rank fusion -------------------------------------------------


def test_fusion_rewards_chunk_ranked_well_by_both_methods():
    a, b, c = _chunk(idx=0), _chunk(idx=1), _chunk(idx=2)
    vector_ranking = [a, b, c]
    keyword_ranking = [b, c, a]

    fused = _fuse_rankings([vector_ranking, keyword_ranking])

    # b is #2 and #1 -> beats a (#1 and #3).
    assert [ch["chunk_index"] for ch in fused][0] == 1


def test_fusion_includes_chunks_found_by_only_one_method():
    only_vector = _chunk(idx=0)
    only_keyword = _chunk(idx=1, vector_similarity=0.0)

    fused = _fuse_rankings([[only_vector], [only_keyword]])

    assert {ch["chunk_index"] for ch in fused} == {0, 1}


def test_fusion_dedupes_same_chunk_across_rankings():
    same = _chunk(idx=7)
    fused = _fuse_rankings([[same], [same]])
    assert len(fused) == 1


def test_fusion_score_matches_rrf_formula():
    chunk = _chunk(idx=0)
    fused = _fuse_rankings([[chunk], [chunk]], k=60)
    # rank 1 in both lists: 1/61 + 1/61
    assert fused[0]["similarity"] == pytest.approx(2 / 61)


def test_fusion_preserves_best_vector_similarity_for_abstention():
    from_vector = _chunk(idx=0, similarity=0.8, vector_similarity=0.8)
    from_keyword = _chunk(idx=0, similarity=12.3, vector_similarity=0.0)

    fused = _fuse_rankings([[from_vector], [from_keyword]])

    # The real cosine signal must survive fusion, not be overwritten
    # by the keyword list's 0.0 placeholder.
    assert fused[0]["vector_similarity"] == 0.8


def test_fusion_distinguishes_same_index_in_different_sources():
    a = _chunk(source="a.pdf", idx=0)
    b = _chunk(source="b.pdf", idx=0)
    fused = _fuse_rankings([[a, b]])
    assert len(fused) == 2


# --- naive reranker ---------------------------------------------------------


def test_lexical_rerank_promotes_chunk_with_more_query_terms():
    weak = _chunk(idx=0, text="general market commentary", similarity=0.9)
    strong = _chunk(idx=1, text="operating margin rose in Q3", similarity=0.5)

    reranked = _lexical_rerank("Q3 operating margin", [weak, strong])

    assert reranked[0]["chunk_index"] == 1


def test_lexical_rerank_breaks_ties_by_existing_score():
    low = _chunk(idx=0, text="no overlap here", similarity=0.2)
    high = _chunk(idx=1, text="nothing matching", similarity=0.7)

    reranked = _lexical_rerank("revenue", [low, high])

    assert reranked[0]["chunk_index"] == 1


# --- BM25 -------------------------------------------------------------------


def test_bm25_search_ranks_exact_term_match_first(monkeypatch):
    corpus = [
        {"id": "a", "text": "general discussion of the economy", "page_number": 1, "chunk_index": 0, "source": "r.pdf"},
        {"id": "b", "text": "EBITDA margin was 18% in fiscal 2025", "page_number": 2, "chunk_index": 1, "source": "r.pdf"},
        {"id": "c", "text": "the board met twice", "page_number": 3, "chunk_index": 2, "source": "r.pdf"},
    ]
    monkeypatch.setattr(R, "get_all_chunks", lambda source=None: corpus)

    results = R._bm25_search("EBITDA margin", fetch_k=5, source=None)

    assert results[0]["chunk_index"] == 1


def test_bm25_search_drops_chunks_with_no_matching_terms(monkeypatch):
    corpus = [
        {"id": "a", "text": "revenue grew", "page_number": 1, "chunk_index": 0, "source": "r.pdf"},
        {"id": "b", "text": "unrelated text", "page_number": 1, "chunk_index": 1, "source": "r.pdf"},
        {"id": "c", "text": "more unrelated words", "page_number": 1, "chunk_index": 2, "source": "r.pdf"},
    ]
    monkeypatch.setattr(R, "get_all_chunks", lambda source=None: corpus)

    results = R._bm25_search("revenue", fetch_k=5, source=None)

    assert [r["chunk_index"] for r in results] == [0]


def test_bm25_search_empty_corpus_returns_empty(monkeypatch):
    monkeypatch.setattr(R, "get_all_chunks", lambda source=None: [])
    assert R._bm25_search("anything", fetch_k=5, source=None) == []


# --- retrieve(): flag routing -----------------------------------------------


@pytest.fixture
def fake_vector(monkeypatch):
    """Vector search that returns chunk 0 first, then 1, 2... and records fetch_k."""
    calls = {}

    def _fake(query, fetch_k, source):
        calls["fetch_k"] = fetch_k
        return [_chunk(idx=i, text=f"vector chunk {i}", similarity=0.9 - i * 0.1) for i in range(fetch_k)]

    monkeypatch.setattr(R, "_vector_search", _fake)
    return calls


def test_retrieve_default_is_vector_only_and_fetches_exactly_top_k(flags, fake_vector, monkeypatch):
    flags(hybrid_retrieval=False, rerank=False)

    def _bm25_should_not_run(*a, **kw):
        raise AssertionError("BM25 ran with hybrid_retrieval=False")

    monkeypatch.setattr(R, "_bm25_search", _bm25_should_not_run)

    results = R.retrieve("q", top_k=3)

    assert len(results) == 3
    assert fake_vector["fetch_k"] == 3


def test_retrieve_hybrid_widens_pool_and_runs_bm25(flags, fake_vector, monkeypatch):
    flags(hybrid_retrieval=True, rerank=False)
    bm25_called = {}

    def _fake_bm25(query, fetch_k, source):
        bm25_called["yes"] = True
        return []

    monkeypatch.setattr(R, "_bm25_search", _fake_bm25)

    results = R.retrieve("q", top_k=3)

    assert bm25_called.get("yes")
    assert fake_vector["fetch_k"] == R.CANDIDATE_POOL
    assert len(results) == 3


def test_retrieve_rerank_flag_applies_reranker(flags, fake_vector, monkeypatch):
    flags(hybrid_retrieval=False, rerank=True)
    reranked = {}

    def _fake_rerank(query, chunks):
        reranked["n"] = len(chunks)
        return list(reversed(chunks))

    monkeypatch.setattr(R, "_lexical_rerank", _fake_rerank)

    results = R.retrieve("q", top_k=2)

    assert reranked["n"] == R.CANDIDATE_POOL
    # Reversed pool, then truncated: last candidate first.
    assert results[0]["chunk_index"] == R.CANDIDATE_POOL - 1
