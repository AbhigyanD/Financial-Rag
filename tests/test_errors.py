"""Tests for the translate_errors helper used by api.py routes."""

import pytest
from fastapi import HTTPException

from financial_rag.errors import translate_errors


class FakeStorageError(Exception):
    pass


class FakeEmbeddingError(Exception):
    pass


def test_translate_errors_maps_matched_exception_to_http_exception():
    with pytest.raises(HTTPException) as exc_info:
        with translate_errors((FakeStorageError, 500, "Failed to store")):
            raise FakeStorageError("disk full")

    assert exc_info.value.status_code == 500
    assert "Failed to store: disk full" in exc_info.value.detail


def test_translate_errors_picks_the_matching_mapping():
    with pytest.raises(HTTPException) as exc_info:
        with translate_errors(
            (FakeStorageError, 500, "Failed to store"),
            (FakeEmbeddingError, 502, "Failed to embed"),
        ):
            raise FakeEmbeddingError("rate limited")

    assert exc_info.value.status_code == 502
    assert "Failed to embed: rate limited" in exc_info.value.detail


def test_translate_errors_lets_unmapped_exceptions_propagate():
    with pytest.raises(ValueError):
        with translate_errors((FakeStorageError, 500, "Failed to store")):
            raise ValueError("unrelated bug")


def test_translate_errors_does_nothing_on_success():
    with translate_errors((FakeStorageError, 500, "Failed to store")):
        result = 1 + 1

    assert result == 2
