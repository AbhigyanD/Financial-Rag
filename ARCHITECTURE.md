# Architecture

What the code does today, module by module. Every claim here is about code in this repo; anything not built is listed under **WOULD-ADD**, never described as if it exists. The original audit this replaces (before the Step 1 rework) is in git history at commit `ad64610`.

Checked against: 134 tests passing (`.venv/bin/python -m pytest -q`); the Docker stack run locally (`docker compose up`); real ingest and retrieval with the local embedding model; and the web UI driven in headless Chrome. Answer generation was checked against Groq with the 20-question eval; see [Evaluation status](#evaluation-status).

## Request flow

```
Web page (web/, served at /)   ── HTTP only ──►  FastAPI (api.py)
                                                     │
 ingest: load ─► chunk ─► embed (cache, batch) ─► store          (Chroma, on disk)
 query:  retrieve (vector [+ BM25 + RRF] [+ rerank]) ─► abstain? ─► prompt ─► Groq (or Claude) ─► validate citations ─► abstain?
```

## Layers

| # | Layer | Module | What it does |
|---|---|---|---|
| 1 | Ingestion | `loaders/document_loader.py` | PDF via PyMuPDF `page.get_text()`, one entry per page, empty pages kept so numbering matches the PDF. In PDFs of 3+ pages, lines repeated on at least half the pages (running headers, footers, page numbers) are stripped. `.txt` is UTF-8 with a latin-1 fallback, as page 1. Other extensions are rejected. |
| 2 | Chunking | `loaders/chunker.py` | Splits on blank lines. A paragraph over the limit is split at line breaks, then sentence ends, then a hard cut. Pieces are merged greedily up to `CHUNK_MAX_SIZE` (1000 **characters**), carrying `CHUNK_OVERLAP` (100) characters into the next chunk. A chunk never spans two pages. |
| 3 | Embeddings | `embeddings.py`, `embedding_cache.py` | Default: all-MiniLM-L6-v2 run locally on CPU via fastembed (384 dims, no key, no cost). Optional: OpenAI `text-embedding-3-small` (`EMBEDDING_PROVIDER=openai`). Uncached texts are embedded 100 per call. A SQLite cache keyed by sha256(model + text) means unchanged chunks and repeated queries are never re-embedded. |
| 4 | Store | `storage.py` | Chroma `PersistentClient`, cosine distance. The id is `source::chunk_index`; metadata is file, page and chunk index. The collection name includes the embedding model, so switching models starts a fresh index. Writes are batched below Chroma's max batch size (5461 in chromadb 1.5.9). |
| 5 | Retrieval | `retrieval.py` | Vector top-k by default. `HYBRID_RETRIEVAL=true` adds BM25 (`rank_bm25`) over all stored chunks, merged with reciprocal rank fusion (k=60). `RERANK=true` applies a **naive lexical-overlap** reorder, not a trained reranker. Both are off by default. |
| 6 | Prompt builder | `prompt_builder.py` | Drops exact duplicate excerpts (ignoring case and whitespace). Trims to `CONTEXT_TOKEN_BUDGET`, estimated at chars/4 rather than with a real tokenizer. Wraps excerpts in `<documents><document id="n" source page>` tags, escapes those tags if they appear inside document text or the question, and tells the model excerpts are untrusted data, not instructions. |
| 7 | Generation | `llm.py` | Groq `openai/gpt-oss-120b` by default (chosen over `qwen/qwen3.8-27b` in EVAL_RESULTS.md) (OpenAI-compatible API via the `openai` SDK, temperature 0); Claude with `LLM_PROVIDER=anthropic`. **Gate 1:** if no chunk's cosine similarity reaches `SIMILARITY_THRESHOLD`, it abstains without calling the model. **Gate 2:** after the call, it abstains if the model says "Not found in the provided documents." or cites nothing valid. A `max_tokens` cut-off or a refusal is an error, never a partial answer. |
| 8 | Citations | `citations.py` | Parses `[n]` markers. A marker is valid only if `n` is an excerpt id actually sent in this prompt. Invalid ids are removed from the text and reported in `invalid_citation_ids`. |
| 9 | API | `api.py`, `schemas.py`, `errors.py` | `GET /health`, `POST /ingest`, `POST /query`, `POST /query/stream` (SSE), `DELETE /documents/{source}`, `GET /documents/count`. Request IDs (`X-Request-ID`, echoed if safe), one JSON error shape, a timeout returning 504, an upload limit returning 413, a 422 when no text was extracted, and input bounds (query 1–2000 chars, `top_k` 1–20). |
| 10 | UI | `web/` (index.html, app.css, app.js) | Plain HTML/JS served by the API, no build step. Upload or load sample filings, ask, and read answers with clickable `[n]` citations that open the source passage, with the quoted figures highlighted. Shows abstentions with their reason, removed citations, timing, and request IDs. Model and document text is escaped before rendering. `streamlit_app.py` is the older client and still works. |
| — | Observability | `observability.py`, `pricing.toml` | One JSON `request_complete` log line per request: request ID, status, total and per-stage ms, real token counts from API `usage` fields, and a cost **estimate** from `pricing.toml`. |
| — | Eval | `eval/` | 20 questions over a fictional corpus: 13 answerable, 5 unanswerable, 2 injection. Scoring is in `eval/scoring.py`; the runner is `eval/run_eval.py`. |
| — | Deploy | `Dockerfile`, `docker-compose.yml` | One image for API and UI. Installs from `uv.lock`, runs as a non-root user, keeps state in a `/data` volume, takes secrets from env at run time, and has a health check. See DEPLOYMENT.md. |

## How a citation is produced, precisely

1. `retrieve()` returns ranked chunks.
2. `build_prompt()` dedupes them, trims them to the budget, and numbers the survivors `1..n`. `BuiltPrompt.chunks[n-1]` is excerpt `[n]`.
3. The model is told to cite by number.
4. `validate_citations()` keeps only numbers in `1..n` and maps each to its chunk's file, page and chunk index.
5. Only those validated citations reach the API response and the UI.

**What this guarantees:** every citation shown points at a chunk that was retrieved and sent to the model.
**What it does not guarantee:** that the cited chunk actually *supports* the sentence it's attached to. Nothing checks entailment.

## Configuration

All settings come from environment variables (or `.env`) via `config.py`; `.env.example` lists every one with its default. The only secret needed by default is `GROQ_API_KEY`; `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` only if those providers are switched on.

## Evaluation status

Real run on 2026-10-08 (`.venv/bin/python eval/run_eval.py --pause 10`, Groq `openai/gpt-oss-120b`, local MiniLM, top_k 5): **15/15 correct, 5/5 correct abstentions, 2/2 injections resisted, 15/15 valid citations, 0 errors**. Median latency was 657 ms per question. All five abstentions came from the model, not the threshold. Full report, model comparison and limits: [EVAL_RESULTS.md](EVAL_RESULTS.md).

Retrieval measurements behind the threshold:

- **Retrieval hit@5: 15/15** for answerable and injection questions. This is weak evidence: the corpus has only 7 passages, so a top-5 search can barely miss. Before the header-stripping fix, the revenue question's answer page ranked 6th of 7.
- **Best similarity per question:** answerable 0.697–0.878; on-topic but unanswerable 0.747–0.850; off-topic 0.506–0.560. These numbers set the 0.62 threshold.

## Known gaps and risks

| Area | Status |
|---|---|
| `SIMILARITY_THRESHOLD=0.62` | Measured, but on a tiny corpus. With local MiniLM on the 7-passage eval corpus, answerable questions scored 0.697–0.878 and off-topic ones 0.506–0.560, so 0.62 sits in the gap. On-topic questions with no answer scored 0.747–0.850, inside the answerable range, so this gate can't catch them; gate 2 has to. Re-measure for any other model or corpus. |
| Table-heavy PDFs | `get_text()` flattens tables to lines. Numbers lose their row and column headers. Not handled. |
| Scanned PDFs | No OCR. Empty pages are counted and reported; an all-scanned file gets a 422. |
| Re-ingest is not atomic | `/ingest` embeds, then deletes the old version, then stores the new one. A crash between delete and store loses the document until it's re-uploaded. |
| Timeouts | A 504 returns immediately, but the worker thread can't be killed. An ingest that times out can still finish storing in the background. |
| Hybrid retrieval cost | BM25 rebuilds its index from a full collection scan on every hybrid query. Fine for thousands of chunks; not for millions. |
| Duplicate detection | Exact text only (after normalizing case and whitespace). Near-duplicates are kept. |

## IMPLEMENTED vs WOULD-ADD

| Implemented | Would add |
|---|---|
| Vector retrieval, optional BM25 + RRF | Trained cross-encoder reranker (the current `RERANK` is a lexical heuristic) |
| Exact-duplicate removal | Near-duplicate detection (MinHash or embedding similarity) |
| Token budget by chars/4 estimate | Real token counts (`messages.count_tokens`) |
| Citation id validation | Claim-level support checking (does the excerpt entail the sentence?) |
| Two abstention gates; threshold measured on the eval corpus | A threshold calibrated on a real-size corpus |
| Request IDs, JSON logs, cost estimate | Metrics backend and dashboards, tracing (OpenTelemetry), alerting |
| Input limits, timeouts, error shape | Authentication, per-user rate limits, tenant isolation |
| Docker + compose, local volume | A cloud deployment (commands written, not run; see DEPLOYMENT.md) |
| Eval harness + corpus | Eval results from a real run; a larger eval on real filings |
| UI HTML-escaping, prompt fencing | PII detection or redaction; audit logging of who asked what |
