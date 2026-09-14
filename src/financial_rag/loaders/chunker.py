"""Split extracted text into chunks while preserving page metadata.

This is stage 2 of the ingest pipeline: given a list[PageText] from stage 1
(document_loader), split the text into smaller chunks suitable for embedding.
Each chunk retains its source page number for later citations.

Chunking strategies:
- By character count: simple, fast, predictable size
- By sentence/paragraph: respects document structure, fewer boundary breaks
- By token count: precise for a given model's context window (requires tokenizer)

For now, we'll focus on character-based chunking with paragraph-aware splitting.
"""

from __future__ import annotations

import re
from typing import TypedDict

from financial_rag.loaders.document_loader import PageText


class Chunk(TypedDict):
    """One chunk of text with its source page number."""

    page_number: int
    text: str
    chunk_index: int  # 0-indexed position in the sequence


class ChunkingError(Exception):
    """Raised when chunking fails."""


def _split_by_paragraphs(text: str) -> list[str]:
    """Split text on paragraph boundaries (blank lines).

    Preserves the text within each paragraph but removes the blank lines.
    Returns a list of non-empty paragraphs.
    """
    split_paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in split_paragraphs if p.strip()]

def _merge_chunks(
    paragraphs: list[str], max_chunk_size: int
) -> list[str]:
    """Greedily merge paragraphs until reaching max_chunk_size.

    Combines multiple paragraphs into one chunk if they fit within
    max_chunk_size (in characters). Once adding the next paragraph would
    exceed the limit, start a new chunk.

    Returns a list of merged chunks. Each chunk is at most max_chunk_size
    characters (unless a single paragraph exceeds it, in which case the
    chunk will be larger).
    """
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if not current_chunk:
            current_chunk = paragraph
        else:
            potential_chunk = current_chunk + "\n\n" + paragraph
            if len(potential_chunk) <= max_chunk_size:
                current_chunk = potential_chunk
            else:
                chunks.append(current_chunk)
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)  # Add the last chunk if non-empty
    return chunks

def chunk_pages(
    pages: list[PageText], max_chunk_size: int = 1000, overlap: bool = False
) -> list[Chunk]:
    """Split a list of pages into chunks while preserving page numbers.

    Args:
        pages: List of PageText dicts from document_loader.
        max_chunk_size: Target chunk size in characters. Chunks may exceed
            this if a single paragraph is larger.
        overlap: Not yet implemented; for future use (e.g., sliding window).

    Returns:
        A list of Chunk dicts, each with page_number, text, and chunk_index.

    Strategy:
        1. For each page, split its text into paragraphs.
        2. Merge paragraphs within a page until reaching max_chunk_size.
        3. Stamp each chunk with the source page_number.
        4. Assign a global chunk_index (0, 1, 2, ...).
    """
    chunks = []
    chunk_index = 0
    for page in pages:
        page_number = page["page_number"]
        text = page["text"]
        paragraphs = _split_by_paragraphs(text)
        merged_chunks = _merge_chunks(paragraphs, max_chunk_size)
        for chunk_text in merged_chunks:
            chunk = Chunk(
                page_number=page_number,
                text=chunk_text,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)
            chunk_index += 1
    return chunks


def count_chunks(pages: list[PageText], max_chunk_size: int = 1000) -> int:
    """Estimate how many chunks will be produced without actually chunking."""
    chunks = chunk_pages(pages, max_chunk_size)
    return len(chunks)
