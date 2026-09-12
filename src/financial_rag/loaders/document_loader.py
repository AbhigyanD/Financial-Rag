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
    # TODO: handle three cases:
    #   1. already bytes/bytearray -> return as-is (cast to bytes)
    #   2. has a .read() method (UploadFile, Streamlit upload, open file) ->
    #      read it, then try to .seek(0) so the caller can re-read the
    #      stream later (wrap the seek in try/except OSError/ValueError)
    #   3. otherwise assume it's a filesystem path -> open(..., "rb").read()
    raise NotImplementedError


def _clean_text(text: str) -> str:
    """Collapse obvious PDF-extraction noise without touching real structure.

    Only squash runs of 3+ blank lines down to a single paragraph
    break. Don't strip anything else -- chunking still needs real
    paragraph breaks to split on.
    """
    # TODO: re.sub a 3+-newline run down to "\n\n", then strip leading/
    # trailing newlines.
    raise NotImplementedError


def load_pdf(file_path_or_bytes: FileInput) -> list[PageText]:
    """Extract text from a PDF, one entry per page (1-indexed)."""
    # TODO:
    #   1. raw = _read_bytes(file_path_or_bytes)
    #   2. try: fitz.open(stream=raw, filetype="pdf")
    #      except Exception -> raise DocumentLoadError, chaining with `from exc`
    #   3. loop `for page_num, page in enumerate(doc)`, call page.get_text()
    #      (wrap in its own try/except -> DocumentLoadError naming the page)
    #   4. append {"page_number": page_num + 1, "text": _clean_text(text)}
    #      -- keep the entry even if text is "" (scanned/no-OCR page), so
    #      page numbers stay aligned with the real PDF page count
    #   5. doc.close() in a finally block
    raise NotImplementedError


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
