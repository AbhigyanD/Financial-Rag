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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from financial_rag.config import settings
from financial_rag.embeddings import EmbeddingError, embed_chunks
from financial_rag.errors import translate_errors
from financial_rag.llm import LLMError, generate_answer
from financial_rag.loaders.chunker import chunk_pages
from financial_rag.loaders.document_loader import DocumentLoadError, load_document
from financial_rag.retrieval import RetrievalError, retrieve
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

    citations = [
        CitationResponse(
            source=c["source"],
            page_number=c["page_number"],
            similarity=c["similarity"],
            text=c["text"],
        )
        for c in answer["citations"]
    ]

    return QueryResponse(answer=answer["text"], citations=citations)
