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

Run locally with:
    uv run uvicorn financial_rag.api:app --reload
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from financial_rag.config import settings
from financial_rag.embeddings import EmbeddingError, embed_chunks
from financial_rag.llm import LLMError, generate_answer
from financial_rag.loaders.chunker import chunk_pages
from financial_rag.loaders.document_loader import DocumentLoadError, load_document
from financial_rag.retrieval import RetrievalError, retrieve
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


class IngestResponse(BaseModel):
    source: str
    pages: int
    chunks_stored: int


class DeleteResponse(BaseModel):
    source: str
    chunks_deleted: int


class CountResponse(BaseModel):
    total_chunks: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    source: Optional[str] = None


class CitationResponse(BaseModel):
    source: str
    page_number: int
    similarity: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a document and run it through stages 1-4 of the pipeline."""
    filename = file.filename or "uploaded_file"
    contents = await file.read()

    try:
        pages = load_document(contents, filename=filename)
    except DocumentLoadError as e:
        raise HTTPException(status_code=400, detail=f"Failed to load document: {e}") from e

    chunks = chunk_pages(pages)

    try:
        embedded_chunks = embed_chunks(chunks)
    except EmbeddingError as e:
        raise HTTPException(status_code=502, detail=f"Failed to embed document: {e}") from e

    try:
        chunks_stored = store_chunks(embedded_chunks, source=filename)
    except StorageError as e:
        raise HTTPException(status_code=500, detail=f"Failed to store document: {e}") from e

    return IngestResponse(source=filename, pages=len(pages), chunks_stored=chunks_stored)


@app.delete("/documents/{source}", response_model=DeleteResponse)
def remove_document(source: str) -> DeleteResponse:
    """Delete all chunks belonging to a previously ingested document."""
    try:
        chunks_deleted = delete_source(source)
    except StorageError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}") from e

    return DeleteResponse(source=source, chunks_deleted=chunks_deleted)


@app.get("/documents/count", response_model=CountResponse)
def documents_count() -> CountResponse:
    """Return the total number of chunks currently indexed."""
    try:
        total = count_stored_chunks()
    except StorageError as e:
        raise HTTPException(status_code=500, detail=f"Failed to count chunks: {e}") from e

    return CountResponse(total_chunks=total)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Ask a question and get a cited answer, running stages 5-6."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        chunks = retrieve(request.query, top_k=request.top_k, source=request.source)
    except RetrievalError as e:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e}") from e

    try:
        answer = generate_answer(request.query, chunks)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {e}") from e

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
