"""Tests for embeddings.py's batching and cache behavior (mocked OpenAI client).

No real API calls — a fake client records how many times it was called
and with what, so these assert the actual cost-saving behavior (real
batching, cache hits) rather than just "it returns something."
"""

from unittest.mock import MagicMock, patch

import pytest

import financial_rag.embeddings as E


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Each test gets its own empty on-disk cache, so tests can't leak
    cached vectors into each other via a shared ./embedding_cache/ dir.

    Settings is a frozen dataclass (by design — see config.py), so we
    can't monkeypatch an attribute on the shared `settings` instance;
    instead patch the function that resolves the cache file path.
    """
    db_path = str(tmp_path / "embeddings.sqlite3")
    monkeypatch.setattr("financial_rag.embedding_cache._cache_path", lambda: db_path)


def _fake_client(vector_len: int = 3) -> MagicMock:
    def fake_create(input, model):
        texts = input if isinstance(input, list) else [input]
        return MagicMock(data=[MagicMock(embedding=[0.1] * vector_len) for _ in texts])

    client = MagicMock()
    client.embeddings.create.side_effect = fake_create
    return client


def _chunks(*texts: str) -> list[dict]:
    return [
        {"page_number": 1, "text": text, "chunk_index": i} for i, text in enumerate(texts)
    ]


def test_embed_chunks_makes_one_api_call_per_batch_not_per_chunk():
    client = _fake_client()
    with patch.object(E, "_get_client", return_value=client):
        E.embed_chunks(_chunks("a", "b", "c"), batch_size=100)

    # Real batching: 3 chunks, batch_size=100 => all 3 in ONE call.
    assert client.embeddings.create.call_count == 1
    call_args = client.embeddings.create.call_args
    assert call_args.kwargs["input"] == ["a", "b", "c"]


def test_embed_chunks_splits_into_multiple_batches():
    client = _fake_client()
    with patch.object(E, "_get_client", return_value=client):
        E.embed_chunks(_chunks("a", "b", "c", "d", "e"), batch_size=2)

    # 5 chunks, batch_size=2 => 3 calls (2, 2, 1).
    assert client.embeddings.create.call_count == 3


def test_embed_chunks_second_call_hits_cache_not_api():
    client = _fake_client()
    with patch.object(E, "_get_client", return_value=client):
        E.embed_chunks(_chunks("x", "y"), batch_size=100)
        assert client.embeddings.create.call_count == 1

        # Same text again — should be served entirely from cache.
        E.embed_chunks(_chunks("x", "y"), batch_size=100)
        assert client.embeddings.create.call_count == 1


def test_embed_chunks_only_calls_api_for_new_text():
    client = _fake_client()
    with patch.object(E, "_get_client", return_value=client):
        E.embed_chunks(_chunks("x", "y"), batch_size=100)
        E.embed_chunks(_chunks("x", "y", "z"), batch_size=100)

    # Second call: "x" and "y" are cached, only "z" should hit the API.
    assert client.embeddings.create.call_count == 2
    second_call_input = client.embeddings.create.call_args_list[1].kwargs["input"]
    assert second_call_input == ["z"]


def test_embed_chunks_deduplicates_identical_text_within_one_call():
    client = _fake_client()
    with patch.object(E, "_get_client", return_value=client):
        E.embed_chunks(_chunks("same", "same", "same"), batch_size=100)

    # Three chunks with identical text should only be embedded once.
    assert client.embeddings.create.call_args.kwargs["input"] == ["same"]


def test_embed_chunks_preserves_order_and_chunk_metadata():
    client = _fake_client()
    with patch.object(E, "_get_client", return_value=client):
        result = E.embed_chunks(_chunks("a", "b"), batch_size=100)

    assert [c["text"] for c in result] == ["a", "b"]
    assert [c["chunk_index"] for c in result] == [0, 1]
    assert all(len(c["embedding"]) == 3 for c in result)


def test_embed_chunks_empty_list_makes_no_api_call():
    client = _fake_client()
    with patch.object(E, "_get_client", return_value=client):
        result = E.embed_chunks([], batch_size=100)

    assert result == []
    client.embeddings.create.assert_not_called()


def test_embed_text_empty_string_raises_without_calling_api():
    client = _fake_client()
    with patch.object(E, "_get_client", return_value=client):
        with pytest.raises(E.EmbeddingError, match="empty"):
            E.embed_text("   ")

    client.embeddings.create.assert_not_called()
