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


def test_page_with_no_blank_lines_is_still_split_to_max_size():
    # PyMuPDF often returns a page with single newlines only — one
    # "paragraph". It must not become one oversized chunk.
    lines = "\n".join(f"Line {i} of a page that lost its blank lines." for i in range(60))
    chunks = chunk_pages([{"page_number": 3, "text": lines}], max_chunk_size=300)

    assert len(chunks) > 1
    assert all(len(c["text"]) <= 300 for c in chunks)
    assert all(c["page_number"] == 3 for c in chunks)


def test_oversized_text_without_line_breaks_splits_on_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(50))
    chunks = chunk_pages([{"page_number": 1, "text": text}], max_chunk_size=200)

    assert all(len(c["text"]) <= 200 for c in chunks)
    assert all(c["text"].endswith(".") for c in chunks)


def test_unbreakable_text_is_hard_cut_at_max_size():
    chunks = chunk_pages([{"page_number": 1, "text": "x" * 950}], max_chunk_size=400)
    assert [len(c["text"]) for c in chunks] == [400, 400, 150]


def test_no_text_is_lost_when_splitting_oversized_pages():
    words = [f"w{i}" for i in range(400)]
    text = " ".join(f"{w}." for w in words)
    chunks = chunk_pages([{"page_number": 1, "text": text}], max_chunk_size=120)
    joined = " ".join(c["text"] for c in chunks)
    assert all(f"{w}." in joined for w in words)


def test_empty_and_whitespace_pages_produce_no_chunks():
    assert chunk_pages([{"page_number": 1, "text": ""}, {"page_number": 2, "text": "  \n\n "}]) == []


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
