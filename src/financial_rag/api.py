"""FastAPI interface for the Financial RAG pipeline.

This is stage 7: exposes the full pipeline (stages 1-6) over HTTP so a
frontend (or curl/Postman) can upload documents and ask questions.

Endpoints:
    POST /documents        Upload a PDF/TXT file — runs stages 1-4
                            (load -> chunk -> embed -> store).
    DELETE /documents/{source}  Remove a previously ingested document.
    GET  /documents/count  Total chunks currently indexed.
    POST /query             Ask a question — runs stages 5-6
                            (retrieve -> generate answer with citations).
    GET  /health            Liveness check.

Request/response shapes live in schemas.py; exception-to-HTTP-status
mapping lives in errors.py — this file is just route wiring.

Run locally with:
    uv run uvicorn financial_rag.api:app --reload
"""

from __future__ import annotations

import json
from typing import Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from financial_rag.config import settings
from financial_rag.embeddings import EmbeddingError, embed_chunks
from financial_rag.errors import translate_errors
from financial_rag.llm import LLMError, generate_answer, generate_answer_stream
from financial_rag.loaders.chunker import chunk_pages
from financial_rag.loaders.document_loader import DocumentLoadError, load_document
from financial_rag.retrieval import RetrievalError, RetrievedChunk, retrieve
from financial_rag.schemas import (
    CitationResponse,
    CountResponse,
    DeleteResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from financial_rag.storage import StorageError, count_stored_chunks, delete_source, store_chunks

app = FastAPI(
    title="Financial RAG",
    description="Retrieval-augmented Q&A over financial documents.",
)

# Allows a browser-based frontend (e.g. a Streamlit/React app on a
# different origin) to call this API. Configure allowed origins via
# CORS_ORIGINS in .env — defaults to common local dev ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a document and run it through stages 1-4 of the pipeline."""
    filename = file.filename or "uploaded_file"
    contents = await file.read()

    with translate_errors(
        (DocumentLoadError, 400, "Failed to load document"),
        (EmbeddingError, 502, "Failed to embed document"),
        (StorageError, 500, "Failed to store document"),
    ):
        pages = load_document(contents, filename=filename)
        chunks = chunk_pages(pages)
        embedded_chunks = embed_chunks(chunks)
        chunks_stored = store_chunks(embedded_chunks, source=filename)

    return IngestResponse(source=filename, pages=len(pages), chunks_stored=chunks_stored)


@app.delete("/documents/{source}", response_model=DeleteResponse)
def remove_document(source: str) -> DeleteResponse:
    """Delete all chunks belonging to a previously ingested document."""
    with translate_errors((StorageError, 500, "Failed to delete document")):
        chunks_deleted = delete_source(source)

    return DeleteResponse(source=source, chunks_deleted=chunks_deleted)


@app.get("/documents/count", response_model=CountResponse)
def documents_count() -> CountResponse:
    """Return the total number of chunks currently indexed."""
    with translate_errors((StorageError, 500, "Failed to count chunks")):
        total = count_stored_chunks()

    return CountResponse(total_chunks=total)


def _citations_payload(chunks: list[RetrievedChunk]) -> list[dict]:
    """Shared shape used by both /query and /query/stream for citations."""
    return [
        CitationResponse(
            source=c["source"],
            page_number=c["page_number"],
            similarity=c["similarity"],
            text=c["text"],
        ).model_dump()
        for c in chunks
    ]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Ask a question and get a cited answer, running stages 5-6."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    with translate_errors(
        (RetrievalError, 502, "Retrieval failed"),
        (LLMError, 502, "Answer generation failed"),
    ):
        chunks = retrieve(request.query, top_k=request.top_k, source=request.source)
        answer = generate_answer(request.query, chunks)

    return QueryResponse(answer=answer["text"], citations=_citations_payload(answer["citations"]))


def _sse_event(event: str, data) -> str:
    """Format one Server-Sent Event line. `data` is JSON-encoded."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """Like /query, but streams the answer as it's generated (SSE).

    Event sequence:
        citations  — sent once, immediately after retrieval, as a JSON
                     array of citation objects (same shape as /query).
        delta      — sent repeatedly, each a JSON string of one text
                     fragment; concatenate them in order to get the
                     full answer.
        error      — sent instead of further deltas if generation fails
                     partway through; the stream ends after this.
        done       — sent once, after the last delta (on success only).
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    # Retrieval happens before the streaming response starts, so a
    # RetrievalError still becomes a normal HTTPException instead of an
    # error event buried inside an already-started stream.
    with translate_errors((RetrievalError, 502, "Retrieval failed")):
        chunks = retrieve(request.query, top_k=request.top_k, source=request.source)

    def event_stream() -> Iterator[str]:
        yield _sse_event("citations", _citations_payload(chunks))
        try:
            for delta in generate_answer_stream(request.query, chunks):
                yield _sse_event("delta", delta)
        except LLMError as e:
            # The HTTP response has already started (status 200 sent), so
            # the failure has to travel as an SSE event, not an HTTPException.
            yield _sse_event("error", str(e))
            return
        yield _sse_event("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
