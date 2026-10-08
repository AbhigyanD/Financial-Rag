"""Direct tests for embedding_cache.py — the sqlite cache on its own,
independent of the OpenAI client or embeddings.py's batching logic.
"""

import sqlite3

import pytest

from financial_rag import embedding_cache


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Point the cache at a fresh sqlite file for each test."""
    path = str(tmp_path / "embeddings.sqlite3")
    monkeypatch.setattr(embedding_cache, "_cache_path", lambda: path)
    return path


def test_get_many_on_empty_cache_returns_nothing(db_path):
    assert embedding_cache.get_many("model-a", ["hello"]) == {}


def test_get_many_with_no_texts_returns_empty_dict(db_path):
    assert embedding_cache.get_many("model-a", []) == {}


def test_put_then_get_round_trips_vectors(db_path):
    embedding_cache.put_many("model-a", {"hello": [0.1, 0.2], "world": [0.3, 0.4]})

    result = embedding_cache.get_many("model-a", ["hello", "world"])

    assert result == {"hello": [0.1, 0.2], "world": [0.3, 0.4]}


def test_get_many_returns_only_hits_for_partial_match(db_path):
    embedding_cache.put_many("model-a", {"hello": [0.1]})

    result = embedding_cache.get_many("model-a", ["hello", "missing"])

    # Misses are absent, not an error and not a None value.
    assert result == {"hello": [0.1]}


def test_same_text_under_different_model_is_a_miss(db_path):
    # Vectors from different models aren't comparable — the cache must
    # never hand back model-a's vector to a model-b lookup.
    embedding_cache.put_many("model-a", {"hello": [0.1]})

    assert embedding_cache.get_many("model-b", ["hello"]) == {}


def test_put_many_overwrites_existing_entry(db_path):
    embedding_cache.put_many("model-a", {"hello": [0.1]})
    embedding_cache.put_many("model-a", {"hello": [0.9]})

    assert embedding_cache.get_many("model-a", ["hello"]) == {"hello": [0.9]}


def test_put_many_with_empty_dict_is_a_noop(db_path):
    embedding_cache.put_many("model-a", {})

    assert embedding_cache.get_many("model-a", ["anything"]) == {}


def test_cache_persists_across_connections(db_path):
    # Each call opens and closes its own connection — data must survive
    # that, i.e. it's actually committed to disk, not held in memory.
    embedding_cache.put_many("model-a", {"persisted": [1.0]})

    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()

    assert count == 1


def test_key_is_deterministic_and_model_sensitive():
    assert embedding_cache._key("m", "text") == embedding_cache._key("m", "text")
    assert embedding_cache._key("m1", "text") != embedding_cache._key("m2", "text")
    assert embedding_cache._key("m", "a") != embedding_cache._key("m", "b")


def test_key_does_not_collide_across_model_text_boundary():
    # Without a separator, ("ab", "c") and ("a", "bc") would hash the
    # same concatenated bytes. The \0 separator in _key prevents that.
    assert embedding_cache._key("ab", "c") != embedding_cache._key("a", "bc")


def test_unicode_text_round_trips(db_path):
    embedding_cache.put_many("model-a", {"revenue €1.2M — Q3": [0.5]})

    assert embedding_cache.get_many("model-a", ["revenue €1.2M — Q3"]) == {
        "revenue €1.2M — Q3": [0.5]
    }
