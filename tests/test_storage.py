"""Tests for storage.py against a real, throwaway Chroma store (no API calls)."""

import pytest

import financial_rag.storage as S


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "PERSIST_DIRECTORY", str(tmp_path))
    S._get_client.cache_clear()
    yield S
    S._get_client.cache_clear()


def _chunks(n, dims=3):
    return [
        {"page_number": 1 + i // 10, "text": f"chunk {i}", "chunk_index": i, "embedding": [0.1 * (i % 7 + 1)] * dims}
        for i in range(n)
    ]


def test_store_more_chunks_than_chromas_max_batch(store):
    # A single upsert above get_max_batch_size() raises inside Chroma.
    n = store._get_client().get_max_batch_size() + 5
    assert store.store_chunks(_chunks(n), source="big.pdf") == n
    assert store.count_stored_chunks() == n


def test_reingest_with_upsert_does_not_duplicate(store):
    store.store_chunks(_chunks(4), source="r.pdf")
    store.store_chunks(_chunks(4), source="r.pdf")
    assert store.count_stored_chunks() == 4


def test_delete_source_removes_only_that_document(store):
    store.store_chunks(_chunks(3), source="a.pdf")
    store.store_chunks(_chunks(2), source="b.pdf")
    assert store.delete_source("a.pdf") == 3
    assert {c["source"] for c in store.get_all_chunks()} == {"b.pdf"}


def test_query_returns_metadata_needed_for_citations(store):
    store.store_chunks(_chunks(3), source="r.pdf")
    [hit] = store.query_chunks([0.1, 0.1, 0.1], top_k=1)
    assert set(hit) == {"page_number", "text", "chunk_index", "score", "source"}
    assert hit["source"] == "r.pdf"
