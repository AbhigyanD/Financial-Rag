"""Smoke tests for stage 1 (document_loader) and stage 2 (chunker).

These don't hit any external API (unlike embeddings/storage), so they're
safe to run in CI on every push without secrets.
"""

import pymupdf as fitz

from financial_rag.loaders.chunker import chunk_pages
from financial_rag.loaders.document_loader import load_pdf, load_text


def test_load_pdf_extracts_text():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test content")
    pdf_bytes = doc.tobytes()
    doc.close()

    pages = load_pdf(pdf_bytes)

    assert len(pages) == 1
    assert "Test content" in pages[0]["text"]
    assert pages[0]["page_number"] == 1


def test_load_text_reads_plain_file():
    pages = load_text(b"Paragraph one.\n\nParagraph two.")

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Paragraph one." in pages[0]["text"]


def test_chunk_pages_splits_on_paragraphs():
    pages = load_text(b"Paragraph one.\n\nParagraph two.")
    chunks = chunk_pages(pages, max_chunk_size=100)

    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["page_number"] == 1


def test_chunk_pages_respects_max_chunk_size():
    long_paragraph_a = "A" * 60
    long_paragraph_b = "B" * 60
    pages = load_text(f"{long_paragraph_a}\n\n{long_paragraph_b}".encode())

    chunks = chunk_pages(pages, max_chunk_size=100)

    # Two 60-char paragraphs shouldn't merge into a single 100-char chunk.
    assert len(chunks) == 2
    assert [c["chunk_index"] for c in chunks] == [0, 1]


def test_chunk_pages_default_overlap_is_zero():
    # Guards the old (pre-overlap) behavior: no explicit overlap => chunks
    # don't share any carried-over text.
    long_paragraph_a = "A" * 60
    long_paragraph_b = "B" * 60
    pages = load_text(f"{long_paragraph_a}\n\n{long_paragraph_b}".encode())

    chunks = chunk_pages(pages, max_chunk_size=100)

    assert chunks[1]["text"] == long_paragraph_b


def test_chunk_pages_overlap_carries_text_into_next_chunk():
    long_paragraph_a = "A" * 60
    long_paragraph_b = "B" * 60
    long_paragraph_c = "C" * 60
    pages = load_text(
        f"{long_paragraph_a}\n\n{long_paragraph_b}\n\n{long_paragraph_c}".encode()
    )

    chunks = chunk_pages(pages, max_chunk_size=100, overlap=20)

    # Chunk 1 should start with the last 20 chars of chunk 0's content.
    assert chunks[0]["text"][-20:] in chunks[1]["text"]
    # And it should still contain the next paragraph's text too.
    assert long_paragraph_b in chunks[1]["text"]
