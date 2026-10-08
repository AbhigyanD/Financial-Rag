# Architecture — what actually exists today

Written from a line-by-line read of the current repo (commit `845ecb2`, 2026-09-15) and a real run of the test suite. Every claim below is either a fact about the code as written, or a number from a command shown next to it. Nothing here describes a planned feature as if it were built — see **Gaps and flagged issues** at the end for everything that is not yet implemented, and `README.md`'s own "Next Stages" history for the stage-by-stage build order.

## Pipeline, stage by stage

### 1. PDF/text parsing — `src/financial_rag/loaders/document_loader.py`

- PDF: PyMuPDF (`fitz`), text extracted **page by page** via `page.get_text()`. One `PageText = {"page_number": int, "text": str}` per page, **including empty pages** (a scanned page with no OCR layer still gets an entry, so page numbers stay aligned with the real PDF page count — `load_pdf`, lines 66–83).
- Text cleanup: only `re.sub(r"\n{3,}", "\n\n", text)` — collapses 3+ blank lines to one paragraph break. Nothing else is touched, so real paragraph structure survives for stage 2.
- Plain `.txt`: read as UTF-8, falls back to latin-1 on decode failure, returned as a single page (`page_number: 1`).
- Dispatch (`load_document`): routes on file extension (`.pdf` / `.txt`) from a passed-in or inferred filename. Anything else raises `DocumentLoadError`.
- Accepts a path, raw `bytes`, or a file-like object (so FastAPI's `UploadFile` and Streamlit's upload object both work through the same code path).

### 2. Chunking — `src/financial_rag/loaders/chunker.py`

- **Method:** paragraph-based greedy merge, per page. Each page's text is split on blank lines (`re.split(r"\n\s*\n", text)`), then paragraphs are concatenated into a chunk until the next paragraph would push it past `max_chunk_size` — at which point a new chunk starts.
- **Size:** `max_chunk_size=1000` **characters** (not tokens) by default, configurable per call.
- **Overlap:** `chunk_pages()` accepts an `overlap: bool = False` parameter — **it is accepted but never used anywhere in the function body.** Passing `overlap=True` silently does nothing; no sliding window exists. The docstring does say "not yet implemented," so it isn't lying, but it's a live footgun for anyone who assumes passing the flag works.
- Every chunk is stamped with its source `page_number` (inherited from the page it came from — a chunk never spans two pages) and a global `chunk_index` (0-indexed, sequential across the whole document).

### 3. Embedding model — `src/financial_rag/embeddings.py`

- OpenAI `text-embedding-3-small`, 1536 dimensions (`config.py` defaults, overridable via `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` env vars).
- `embed_chunks(chunks, batch_size=100)` splits the chunk list into groups of `batch_size`, but **within each group it still calls `embed_text()` once per chunk in a Python loop** (`embeddings.py:116`) — i.e. one OpenAI API call per chunk, not one batched call per group. `batch_size` currently only controls how often a progress line is printed; it does not reduce the number of API calls. This is a real gap against anything that implies true batch embedding.
- **No cache.** Every call to `embed_chunks` re-embeds every chunk passed to it, even if that exact text was already embedded in a previous ingest. There is no on-disk or in-memory cache keyed by chunk content.
- `embed_query()` is a thin wrapper around `embed_text()` — same model, so query and document vectors are comparable.

### 4. Vector store — `src/financial_rag/storage.py`

- Chroma, `PersistentClient` writing to `./chroma_data/` (configurable via `PERSIST_DIRECTORY`). One collection, `financial_documents`, created with `metadata={"hnsw:space": "cosine"}` so distances land in `[0, 2]`.
- `store_chunks()` builds a deterministic id per chunk (`f"{source}::{chunk_index}"`) and calls `collection.upsert(...)` — re-ingesting the same filename overwrites its old vectors rather than duplicating them. (This is id-based de-duplication of *re-ingests*, not a cache that avoids re-embedding — see stage 3.)
- `query_chunks(query_embedding, top_k, source=None)` does one `collection.query()` call, optionally filtered by `source` via Chroma's `where`.
- `delete_source(source)` and `count_stored_chunks()` exist and are used by the API/UI.

### 5. Retrieval — `src/financial_rag/retrieval.py`

- `retrieve(query, top_k=5, source=None)`: embeds the query, calls `query_chunks`, and converts Chroma's raw cosine distance into a `[0, 1]` similarity score (`similarity = (2 - distance) / 2`, clamped). This conversion is pure vector search — **no keyword/BM25 component, no reciprocal rank fusion, no reranker exist anywhere in the codebase.**
- `top_k` has no server-side upper bound — the Pydantic `QueryRequest.top_k: int = 5` accepts any integer a caller sends; nothing clamps it.

### 6. Prompt construction — `src/financial_rag/llm.py`

- `build_context(chunks)` numbers each retrieved chunk as `[Excerpt N — {source}, page {page_number}]` followed by its text, joined with blank lines. If no chunks were retrieved, it substitutes a literal `"(No relevant context was found for this query.)"` string.
- The full prompt is one user message: `f"Context:\n\n{context}\n\nQuestion: {query}"`. No delimiters beyond the `[Excerpt N — ...]` headers separate different chunks from each other or from the question.
- `SYSTEM_PROMPT` instructs the model to answer only from the provided excerpts, cite `[source, page N]` inline, and say so when the context is insufficient. **It does not explicitly tell the model that chunk text is untrusted data rather than instructions** — there is no "ignore any instructions that appear inside the excerpts" language. This matters directly for the prompt-injection test case in your Step 2 eval plan.
- No token-budget accounting — nothing counts tokens before sending the request or trims context to fit a budget. At current defaults (5 chunks × ≤1000 chars) this is unlikely to blow a context window, but it isn't enforced.
- No deduplication of near-identical chunks across sources.

### 7. How citations are actually produced — `llm.py` + `api.py`

This is the most important thing to be precise about for an interview.

**What happens:** `generate_answer()`/`generate_answer_stream()` return `Answer.citations = chunks` — literally the same list of `RetrievedChunk`s that was handed to the model as context, unmodified. The API's `/query` and `/query/stream` endpoints serialize that list straight into the `citations` field of the response.

**What does *not* happen:** nothing parses Claude's generated answer text to check which `[source, page N]` tags it actually wrote, and nothing cross-references those tags against the chunk list to confirm each citation corresponds to a chunk that was really retrieved. The "citations" shown to the user are "the chunks that were available to the model," not "the chunks the model actually used or cited." If the model writes a citation to a page that wasn't retrieved (a hallucinated citation), there is currently no code path that would catch it.

### 8. API — `src/financial_rag/api.py`, `schemas.py`, `errors.py`

- FastAPI, typed Pydantic request/response models (`schemas.py`), CORS middleware (origins from `CORS_ORIGINS` env var).
- Endpoints: `POST /documents` (ingest), `DELETE /documents/{source}`, `GET /documents/count`, `POST /query`, `POST /query/stream` (Server-Sent Events), `GET /health`.
- Exception→HTTP-status mapping is centralized in `errors.py::translate_errors`, a context manager each route uses declaratively instead of repeating `try/except → HTTPException`.
- **No auth of any kind** on any endpoint. Anyone who can reach the process can ingest, delete, or query — and each query spends real OpenAI + Anthropic API credits.
- **No request IDs, no structured logging, no per-stage latency timing, no token/cost accounting** anywhere in the request path.
- No explicit server-side request timeout on the route handlers themselves (the SDK clients inside have their own default timeouts, but the FastAPI layer doesn't impose one).

### 9. UI — `streamlit_app.py`

- A pure HTTP client — talks to the FastAPI backend via `requests`, no direct imports from the pipeline package. Sidebar: file upload, chunk count, source filter, top-k slider, delete-by-filename. Main: chat interface that consumes `/query/stream`'s SSE events and renders the answer incrementally, with citations in an expander below each message.
- No request-level error surfacing beyond a generic `st.error(...)` — the UI doesn't distinguish "bad request" from "server error" from "network unreachable" for the user.

## A note on the entry point

`pyproject.toml` declares a console script, `financial-rag = "financial_rag:main"`, and `src/financial_rag/__init__.py` defines:
```python
def main() -> None:
    print("Hello from financial-rag!")
```
This is the unmodified placeholder from the initial `uv init` scaffold. It is not wired to anything — running `financial-rag` from the CLI does nothing useful. Worth either removing or pointing at something real before anyone runs it expecting it to do something.

## Verified, not claimed

- **Tests:** `.venv/bin/python -m pytest -v` → **18 passed**, 0 failed, 4.42s (run just now; `uv` is not on this shell's `PATH` in the current environment, so I used the venv's own interpreter directly — worth knowing if you demo this live).
  - Coverage: loaders (`test_loaders.py`, 4 tests — PDF/text extraction, paragraph chunking, max-size respecting), retrieval math (`test_retrieval.py`, 5 tests — the distance→similarity conversion), LLM prompt formatting and streaming (`test_llm.py`, 5 tests — `build_context`, mocked-Anthropic-client streaming and refusal handling), and the `translate_errors` HTTP-error-mapping helper (`test_errors.py`, 4 tests).
  - **Nothing in the test suite touches citation validity or abstention behavior**, because — per the section above — there is no citation-validation code yet to test, and abstention is currently only a prompt instruction, not something the code checks for.
- **No documents are currently ingested** — `./chroma_data/` does not exist on disk in this environment. I have not personally run a real end-to-end ingest-then-query against live OpenAI/Anthropic APIs in this audit; earlier mocked tests (fake embeddings, fake Chroma distances, mocked Anthropic streaming) exercised the wiring but not real model output quality or real citation behavior.
- Line counts (`wc -l`, real): `storage.py` 222, `llm.py` 183, `api.py` 176, `embeddings.py` 156, `document_loader.py` 136, `chunker.py` 115, `retrieval.py` 92, `config.py` 80, `schemas.py` 45, `errors.py` 44 — plus `streamlit_app.py` 294. ~1,545 lines of application code total, not counting tests.

## Gaps and flagged issues (not implemented — nothing below exists in code today)

Ordered roughly by how much they'd matter in an interview conversation:

1. **No code-level citation validation.** Covered above — the single most important gap to be able to speak to honestly.
2. **No prompt-injection defense.** The system prompt doesn't mark retrieved text as untrusted/non-instructional. A chunk containing "ignore previous instructions and..." would reach the model with no guardrail beyond whatever Claude does on its own.
3. **No embedding cache.** Re-ingesting an unchanged document re-embeds every chunk from scratch — real, avoidable OpenAI cost.
4. **"Batch" embedding isn't batched.** `embed_chunks` still makes one API call per chunk; `batch_size` doesn't reduce call count.
5. **No hybrid retrieval.** Pure vector search only — no BM25, no reciprocal rank fusion, no reranker.
6. **No abstention logic in code.** The model is *asked* to say "not found," but nothing checks retrieval similarity scores or the generated text to confirm it actually did, or to force an abstention when scores are weak.
7. **No auth, no rate limiting.** Every endpoint is open; every `/query` call spends real money.
8. **No observability.** No request IDs, no structured logs, no per-stage latency, no token/cost tracking.
9. **No eval harness, no deployment artifacts (Dockerfile/compose), no `eval/questions.jsonl`.**
10. **Two stale docstrings** (`embed_chunks`, `embed_query`, `get_embedding_dimensions` in `embeddings.py`, and the `PERSIST_DIRECTORY` comment in `storage.py`) still read as unresolved `TODO:` blocks even though the functions below them are fully implemented — misleading to a reader skimming the file.
11. **Dead `main()` entry point** — see above.
12. **`top_k` is unbounded** at the API layer — no `Field(le=...)` constraint on the Pydantic model.
13. **Inconsistent client-init error handling** between `embeddings.py` (`_get_client` checks for the API key explicitly before constructing the client) and `llm.py` (`_get_client` wraps the whole constructor call in a bare `except Exception`) — same job, two different styles.

---

I'm stopping here per your Step 0 instruction. Nothing in the code has been changed. Let me know how you want to sequence Steps 1–5 from here — given the real scope (hybrid retrieval + citation validation + an eval harness + observability + a deployment target, each with its own tests), I'd expect this to run well past the 45-minute checkpoint you set, so I'd rather agree on an order and do it in reviewable chunks than try to push it all through at once.
