"""FastAPI layer over the pipeline: /health, /ingest, /query, /query/stream.

Why: transport concerns — request IDs, timeouts, upload limits, one error
shape — live here so the pipeline modules stay plain, testable functions.

Run locally:
    uv run uvicorn financial_rag.api:app --reload
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
import uuid
from typing import Iterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from financial_rag.config import settings
from financial_rag.embeddings import EmbeddingError, embed_chunks
from financial_rag.errors import translate_errors
from financial_rag.llm import Answer, LLMError, generate_answer, generate_answer_stream
from financial_rag.loaders.chunker import chunk_pages
from financial_rag.loaders.document_loader import DocumentLoadError, load_document
from financial_rag.observability import (
    Trace,
    activate,
    add_tokens,
    configure_logging,
    finish,
    log_event,
    note,
    stage,
)
from financial_rag.retrieval import RetrievalError, retrieve
from financial_rag.schemas import (
    CountResponse,
    DeleteResponse,
    ErrorResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from financial_rag.storage import (
    StorageError,
    count_stored_chunks,
    delete_source,
    store_chunks,
)

configure_logging()

app = FastAPI(
    title="Financial RAG",
    description="Retrieval-augmented Q&A over financial documents, with validated citations.",
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# A client-supplied X-Request-ID is reused only if it looks like an id, so
# arbitrary header text never ends up in logs or response headers.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


# --- request IDs and error shape ------------------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign a request ID and open the request's Trace.

    Every request except /health gets one `request_complete` log line.
    Streaming responses finish their own trace after the last event,
    because their body runs after this middleware has returned.
    """
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    if request.url.path == "/health":
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    trace = Trace(request_id=request_id, route=f"{request.method} {request.url.path}")
    request.state.trace = trace
    with activate(trace):
        try:
            response = await call_next(request)
        except Exception as exc:
            log_event("unhandled_error", level=logging.ERROR, request_id=request_id, error=repr(exc))
            finish(trace, 500, error_type=type(exc).__name__)
            raise

    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        finish(trace, response.status_code)
    response.headers["X-Request-ID"] = request_id
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error(request: Request, status: int, message: str) -> JSONResponse:
    request_id = _request_id(request)
    return JSONResponse(
        status_code=status,
        content={"error": {"status": status, "message": message, "request_id": request_id}},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    return _error(request, exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
    )
    return _error(request, 422, f"Invalid request — {problems}")


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Details stay server-side; the client gets the request ID to report.
    return _error(request, 500, "Internal error. Quote the request_id when reporting this.")


async def _with_timeout(fn, *args):
    """Run blocking pipeline code off the event loop, bounded by a timeout.

    Caveat: on timeout the client gets a 504 immediately, but the worker
    thread can't be killed and finishes in the background. The SDK
    clients carry the same timeout, which bounds how long that lasts.
    """
    # Copy the context explicitly so the worker thread sees this request's Trace.
    ctx = contextvars.copy_context()
    try:
        return await asyncio.wait_for(
            run_in_threadpool(ctx.run, fn, *args), timeout=settings.request_timeout_seconds
        )
    except asyncio.TimeoutError:
        note(timed_out=True)
        raise HTTPException(504, f"Timed out after {settings.request_timeout_seconds:.0f}s.")


# --- routes ------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _ingest(contents: bytes, filename: str) -> tuple[int, int, int]:
    with translate_errors(
        (DocumentLoadError, 400, "Failed to load document"),
        (EmbeddingError, 502, "Failed to embed document"),
        (StorageError, 500, "Failed to store document"),
    ):
        with stage("load"):
            pages = load_document(contents, filename=filename)
        empty_pages = sum(1 for p in pages if not p["text"].strip())
        with stage("chunk"):
            chunks = chunk_pages(
                pages, max_chunk_size=settings.chunk_max_size, overlap=settings.chunk_overlap
            )
        note(source=filename, pages=len(pages), empty_pages=empty_pages, upload_bytes=len(contents))
        if not chunks:
            raise HTTPException(
                422,
                f"No extractable text in '{filename}' ({len(pages)} page(s)). "
                "It may be a scanned PDF without an OCR text layer.",
            )
        with stage("embed"):
            embedded = embed_chunks(chunks)
        # Replace, don't merge: if a re-uploaded version has fewer chunks,
        # upsert alone would leave the old version's extra chunks behind.
        # Embedding happens first so a failed embed never deletes good data.
        with stage("store"):
            delete_source(filename)
            stored = store_chunks(embedded, source=filename)
    return len(pages), empty_pages, stored


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, file: UploadFile = File(...)) -> IngestResponse:
    """Upload a PDF/TXT: load -> chunk -> embed -> store."""
    filename = file.filename or "uploaded_file"
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    contents = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb:g} MB upload limit.")

    pages, empty_pages, stored = await _with_timeout(_ingest, contents, filename)
    return IngestResponse(
        source=filename, pages=pages, empty_pages=empty_pages,
        chunks_stored=stored, request_id=_request_id(request),
    )


@app.delete("/documents/{source}", response_model=DeleteResponse)
def remove_document(source: str) -> DeleteResponse:
    with translate_errors((StorageError, 500, "Failed to delete document")):
        deleted = delete_source(source)
    return DeleteResponse(source=source, chunks_deleted=deleted)


@app.get("/documents/count", response_model=CountResponse)
def documents_count() -> CountResponse:
    with translate_errors((StorageError, 500, "Failed to count chunks")):
        total = count_stored_chunks()
    return CountResponse(total_chunks=total)


def _to_response(answer: Answer, request_id: str) -> QueryResponse:
    return QueryResponse(
        answer=answer["text"],
        abstained=answer["abstained"],
        abstain_reason=answer["abstain_reason"],
        citations=answer["citations"],
        invalid_citation_ids=answer["invalid_citation_ids"],
        request_id=request_id,
    )


def _retrieve(body: QueryRequest):
    with translate_errors((RetrievalError, 502, "Retrieval failed")):
        return retrieve(body.query, top_k=body.top_k, source=body.source)


def _record_answer(answer: Answer) -> None:
    add_tokens(llm_input=answer["input_tokens"], llm_output=answer["output_tokens"])
    note(
        abstained=answer["abstained"],
        abstain_reason=answer["abstain_reason"],
        citations_valid=len(answer["citations"]),
        citations_invalid=len(answer["invalid_citation_ids"]),
    )


def _answer(body: QueryRequest) -> Answer:
    note(top_k=body.top_k, query_chars=len(body.query))
    chunks = _retrieve(body)
    with translate_errors((LLMError, 502, "Answer generation failed")), stage("generate"):
        answer = generate_answer(body.query, chunks)
    _record_answer(answer)
    return answer


@app.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Retrieve -> generate -> validate citations (or abstain)."""
    answer = await _with_timeout(_answer, body)
    return _to_response(answer, _request_id(request))


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """Like /query, streamed as Server-Sent Events.

    Events: `delta` (raw text fragment, repeated) -> `final` (the full
    QueryResponse, with validated citations — replaces the deltas), or
    `error` if generation fails after the stream has started.
    """
    # Retrieval runs before streaming starts, so its failures are normal
    # HTTP errors rather than an error event inside a 200 response.
    note(top_k=body.top_k, query_chars=len(body.query), streamed=True)
    chunks = await _with_timeout(_retrieve, body)
    request_id = _request_id(request)
    trace: Trace = request.state.trace
    # The body below runs in worker threads after the middleware returned.
    # Each step runs inside this one captured context so pipeline code
    # still reports into the request's Trace.
    ctx = contextvars.copy_context()

    def events() -> Iterator[str]:
        outcome = {"status": 200}
        t0 = time.perf_counter()
        first_token_ms = None
        gen = generate_answer_stream(body.query, chunks)
        try:
            while (item := ctx.run(next, gen, None)) is not None:
                kind, payload = item
                if kind == "delta":
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - t0) * 1000, 1)
                    yield _sse("delta", payload)
                else:
                    ctx.run(_record_answer, payload)
                    yield _sse("final", _to_response(payload, request_id).model_dump())
        except LLMError as e:
            outcome = {"status": 502, "stream_error": str(e)}
            yield _sse("error", {"message": str(e), "request_id": request_id})
        finally:
            trace.stages_ms["generate"] = round((time.perf_counter() - t0) * 1000, 1)
            finish(trace, outcome.pop("status"), first_token_ms=first_token_ms, **outcome)

    return StreamingResponse(events(), media_type="text/event-stream")
