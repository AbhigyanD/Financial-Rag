"""Tests for the pure logic in retrieval.py (no vector store or API calls)."""

from financial_rag.retrieval import _distance_to_similarity


def test_zero_distance_is_perfect_similarity():
    assert _distance_to_similarity(0.0) == 1.0


def test_max_cosine_distance_is_zero_similarity():
    assert _distance_to_similarity(2.0) == 0.0


def test_mid_distance_is_mid_similarity():
    assert _distance_to_similarity(1.0) == 0.5


def test_similarity_decreases_as_distance_increases():
    assert _distance_to_similarity(0.2) > _distance_to_similarity(0.8)


def test_out_of_range_distance_is_clamped():
    # Defensive: shouldn't happen with a correctly configured cosine
    # collection, but a stray negative or >2 value must not escape [0, 1].
    assert _distance_to_similarity(-0.5) == 1.0
    assert _distance_to_similarity(3.0) == 0.0
