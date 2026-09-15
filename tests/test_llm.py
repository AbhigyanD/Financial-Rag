"""Tests for the pure logic in llm.py (no API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from financial_rag.llm import LLMError, build_context, generate_answer_stream
from financial_rag.retrieval import RetrievedChunk


def _chunk(**overrides) -> RetrievedChunk:
    base = RetrievedChunk(
        page_number=1,
        text="Revenue grew 20% YoY.",
        chunk_index=0,
        source="report.pdf",
        similarity=0.9,
    )
    base.update(overrides)
    return base


def test_build_context_empty_chunks():
    assert "No relevant context" in build_context([])


def test_build_context_labels_source_and_page():
    context = build_context([_chunk()])

    assert "report.pdf" in context
    assert "page 1" in context
    assert "Revenue grew 20% YoY." in context


def test_build_context_numbers_multiple_excerpts():
    chunks = [
        _chunk(text="First fact.", chunk_index=0),
        _chunk(text="Second fact.", chunk_index=1, page_number=2),
    ]
    context = build_context(chunks)

    assert "Excerpt 1" in context
    assert "Excerpt 2" in context
    assert "First fact." in context
    assert "Second fact." in context


def _mock_stream(text_chunks: list[str], stop_reason: str = "end_turn") -> MagicMock:
    """Build a mock for `client.messages.stream(...)`'s context manager,
    matching the shape generate_answer_stream() relies on:
    `with client.messages.stream(...) as stream: stream.text_stream`
    and `stream.get_final_message()`.
    """
    stream_cm = MagicMock()
    entered = stream_cm.__enter__.return_value
    entered.text_stream = iter(text_chunks)
    entered.get_final_message.return_value = MagicMock(stop_reason=stop_reason)
    return stream_cm


def test_generate_answer_stream_yields_text_incrementally():
    with patch("financial_rag.llm._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _mock_stream(["Hello ", "world"])
        mock_get_client.return_value = mock_client

        fragments = list(generate_answer_stream("query", []))

    assert fragments == ["Hello ", "world"]


def test_generate_answer_stream_raises_on_refusal():
    with patch("financial_rag.llm._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _mock_stream(
            ["partial"], stop_reason="refusal"
        )
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMError, match="declined"):
            list(generate_answer_stream("query", []))
