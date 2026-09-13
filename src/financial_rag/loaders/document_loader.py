"""Load raw text out of an uploaded document, page by page.

This is stage 1 of the ingest pipeline: given a PDF or plain-text file,
return the extracted text plus enough structure (page numbers) that
stage 2 (chunking) can stamp each chunk with a page number for later
citations. Pages are kept separate here rather than concatenated into
one blob, because page number is metadata the rest of the pipeline
needs downstream.
"""

from __future__ import annotations

import os
import re
from typing import BinaryIO, TypedDict, Union

import pymupdf as fitz  # `import fitz` is the deprecated spelling as of PyMuPDF 1.24+


class PageText(TypedDict):
    """One page (or, for plain text, the whole document) of extracted text."""

    page_number: int | None
    text: str


class DocumentLoadError(Exception):
    """Raised when a document is corrupt, unsupported, or fails to parse."""


# A path on disk, raw bytes, or a file-like object (UploadFile, Streamlit's
# UploadedFile, an open file handle, ...).
FileInput = Union[str, os.PathLike, bytes, BinaryIO]


def _read_bytes(file_path_or_bytes: FileInput) -> bytes:
    """Normalize a path / bytes / file-like object into raw bytes."""
    if isinstance(file_path_or_bytes, (bytes, bytearray)):
        return bytes(file_path_or_bytes)
    elif hasattr(file_path_or_bytes, "read"):
        # file-like object (UploadFile, Streamlit upload, open file)
        raw = file_path_or_bytes.read()
        try:
            file_path_or_bytes.seek(0)
        except (OSError, ValueError):
            # some streams don't support seek; that's fine, the caller
            # just won't be able to re-read it later
            pass
        return raw
    elif isinstance(file_path_or_bytes, (str, os.PathLike)):
        # filesystem path
        with open(file_path_or_bytes, "rb") as f:
            return f.read()


def _clean_text(text: str) -> str:
    """Collapse obvious PDF-extraction noise without touching real structure.

    Only squash runs of 3+ blank lines down to a single paragraph
    break. Don't strip anything else -- chunking still needs real
    paragraph breaks to split on.
    """
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_pdf(file_path_or_bytes: FileInput) -> list[PageText]:
    """Extract text from a PDF, one entry per page (1-indexed)."""
    raw = _read_bytes(file_path_or_bytes) # read the bytes from the input (path, bytes, or file-like object)
    pages = []
    try:
        doc = fitz.open(stream=raw, filetype="pdf") # open the PDF from bytes
    except Exception as exc:
        raise DocumentLoadError(f"Could not open PDF: {exc}") from exc
 # raise a DocumentLoadError if the PDF cannot be opened
    try:
        for page_num, page in enumerate(doc): # iterate over each page in the PDF
            try:
                text = page.get_text() # extract text from the page
            except Exception as exc:
                raise DocumentLoadError(f"Failed to extract text from page {page_num + 1}") from exc # raise a DocumentLoadError if text extraction fails
            pages.append({"page_number": page_num + 1, "text": _clean_text(text)}) # append the cleaned text and page number to the pages list
    finally:
        doc.close() # ensure the PDF document is closed after processing


def load_text(file_path_or_bytes: FileInput) -> list[PageText]:
    """Read a plain-text file as a single "page", for a uniform shape."""
    # TODO: _read_bytes(...), decode as utf-8 with a latin-1 fallback
    # (raise DocumentLoadError if both fail), return a single-item list
    # with page_number: 1.
    raise NotImplementedError


def load_document(
    file_path_or_bytes: FileInput, filename: str | None = None
) -> list[PageText]:
    """Dispatch to load_pdf/load_text based on file extension.

    `filename` is required when `file_path_or_bytes` is raw bytes or a
    file-like object with no usable name (pass e.g. UploadFile.filename
    explicitly). It's optional when `file_path_or_bytes` is a path, or
    an upload object that already exposes `.filename`/`.name`.
    """
    # TODO:
    #   1. figure out `name`: prefer the explicit filename= arg, then
    #      getattr(..., "filename", None), then getattr(..., "name", None),
    #      then str(file_path_or_bytes) if it's a str/PathLike
    #   2. if no name at all -> raise DocumentLoadError
    #   3. ext = os.path.splitext(name)[1].lower()
    #   4. dispatch ".pdf" -> load_pdf, ".txt" -> load_text,
    #      else -> raise DocumentLoadError(f"Unsupported file type: {ext}")
    raise NotImplementedError


def get_preview(pages: list[PageText], length: int = 500) -> str:
    """Join page text (in order) until `length` chars are available."""
    # TODO: walk pages in order, collecting page["text"] until you've
    # gathered at least `length` chars, join with "\n\n", then slice to
    # `length`.
    raise NotImplementedError


def get_total_length(pages: list[PageText]) -> int:
    """Total character count across all pages."""
    # TODO: sum(len(page["text"]) for page in pages)
    raise NotImplementedError
