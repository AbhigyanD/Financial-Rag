"""Translate pipeline exceptions into HTTP errors for the API layer.

Every pipeline stage raises its own exception type (DocumentLoadError,
EmbeddingError, StorageError, RetrievalError, LLMError). Routes in api.py
need to turn each into an appropriate HTTPException — this was previously
a `try/except ... raise HTTPException(...)` block repeated at every call
site. `translate_errors` collapses that into one declarative mapping per
route, so the route body reads as plain pipeline calls.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from fastapi import HTTPException

# (exception type, HTTP status code, message prefix)
ErrorMapping = tuple[type[Exception], int, str]


@contextmanager
def translate_errors(*mappings: ErrorMapping) -> Iterator[None]:
    """Run a block, converting any matched exception into an HTTPException.

    Usage:
        with translate_errors(
            (DocumentLoadError, 400, "Failed to load document"),
            (EmbeddingError, 502, "Failed to embed document"),
        ):
            pages = load_document(...)
            embedded = embed_chunks(...)

    Exceptions not listed in `mappings` propagate unchanged — they surface
    as FastAPI's default 500 response rather than being silently mapped,
    so genuine bugs aren't mistaken for one of the expected pipeline errors.
    """
    try:
        yield
    except Exception as exc:
        for exc_type, status_code, message in mappings:
            if isinstance(exc, exc_type):
                raise HTTPException(status_code=status_code, detail=f"{message}: {exc}") from exc
        raise
