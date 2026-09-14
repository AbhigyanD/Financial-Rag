"""Tests for the pure logic in llm.py (no API calls)."""

from financial_rag.llm import build_context
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
