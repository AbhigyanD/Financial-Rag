# Interview walkthrough

How this repo works, what it doesn't do, and why. Grounded in the code at the time of writing; file and function names are real. Companion docs: ARCHITECTURE.md (layers), COST.md, DEPLOYMENT.md.

**One-line pitch:** a layered RAG prototype for financial documents whose main design goal is that it never shows an uncited answer and never shows a citation that doesn't point at a retrieved chunk. When it can't support an answer, it says "Not found in the provided documents."

**What I can't claim yet:** answer quality. The eval harness exists, but no run against the real APIs has happened, so I have no accuracy, abstention, or cost numbers from real traffic.

---

## 1. A request, from question to answer

A user types "What was operating income in fiscal 2025?" in the Streamlit app.

1. **UI.** `streamlit_app.py`'s `_stream_query()` POSTs `{"query", "top_k", "source"}` to `/query/stream`. The UI never imports the pipeline; it's HTTP only.
2. **Middleware.** `api.request_id_middleware` reuses the client's `X-Request-ID` if it matches `^[A-Za-z0-9._-]{1,64}$`, otherwise generates 12 hex characters. It creates an `observability.Trace` and activates it in a context variable.
3. **Validation.** `schemas.QueryRequest` strips whitespace and enforces a query of 1–2000 chars and `top_k` of 1–20. A violation gets a 422 in the standard `{"error": {status, message, request_id}}` shape.
4. **Retrieval, before streaming starts** (so failures are ordinary HTTP errors). `api._with_timeout(_retrieve, body)` runs in a worker thread, with the context copied so the trace follows, under `REQUEST_TIMEOUT_SECONDS`.
   - `retrieval.retrieve()` → `_vector_search()` → `embeddings.embed_query()` → `embed_text()`, which checks `embedding_cache.get_many()` first and calls OpenAI only on a miss (`add_tokens(embedding=...)`).
   - `storage.query_chunks()` runs a Chroma query with cosine distance. `_distance_to_similarity()` maps distance to `(2 − d) / 2`, giving a value in [0, 1].
   - If `HYBRID_RETRIEVAL`: `_bm25_search()` builds a BM25 index from `storage.get_all_chunks()`, and `_fuse_rankings()` merges the two rankings with reciprocal rank fusion. The RRF score replaces `similarity`; the real cosine score stays in `vector_similarity`.
   - If `RERANK`: `_lexical_rerank()`, a query-term-overlap heuristic.
5. **Generation.** `llm.generate_answer_stream()`:
   - `retrieval_is_weak()`: if the best `vector_similarity` is below `SIMILARITY_THRESHOLD`, it yields an abstention immediately, with no Claude call.
   - `prompt_builder.build_prompt()`: `dedupe_chunks()`, then `fit_to_budget()` (chars/4 estimate, drops lowest-ranked first, always keeps one), then `_format_documents()` wraps each excerpt in `<document id="n" source="…" page="…">`, with `_neutralize()` escaping any of our tags that appear in document text or the question.
   - `client.messages.stream(...)` yields text deltas, which go to the browser as SSE `delta` events.
   - `_check_stop()`: `refusal` or `max_tokens` becomes an `LLMError`, sent as an SSE `error` event.
   - `finalize_answer()` → `citations.validate_citations()` keeps `[n]` only where `n` is in `1..len(prompt.chunks)`, strips the rest, and abstains if nothing valid remains or the model wrote the abstain sentence.
6. **Final event.** The validated `QueryResponse` is sent as an SSE `final` event. The UI throws away the streamed draft and renders the final text, numbered sources (file, page, similarity, escaped excerpt), or the not-found state with its reason.
7. **Log.** When the stream ends, `observability.finish()` writes one JSON line: request ID, per-stage ms, time to first token, token counts from the API `usage` fields, and a cost estimate from `pricing.toml`.

Ingest follows the same pattern through `POST /ingest`: size check (413), `load_document`, `chunk_pages`, a 422 if no text came out, `embed_chunks` (cache, then batches of 100), `delete_source`, `store_chunks` (batched under Chroma's 5461 limit).

## 2. Implemented vs would-add

| Area | Implemented | Would add |
|---|---|---|
| Parsing | PyMuPDF text per page; TXT | OCR for scanned pages; table extraction that keeps headers |
| Chunking | Paragraph-first, size-bounded, overlap, page-pure | Token-based sizing; section-aware splitting (headings) |
| Embeddings | Batched, SQLite cache, timeouts | Rate-limit backoff tuning; async batching |
| Retrieval | Vector; optional BM25+RRF; naive rerank flag | Trained cross-encoder reranker; metadata filters beyond source |
| Prompt | Dedupe, budget, delimiters, untrusted-data framing | Real token counting; near-duplicate removal |
| Generation | Two abstention gates; truncation and refusal errors | Calibrated threshold; effort tuning |
| Citations | Id validation against sent chunks | Checking the excerpt supports the sentence (entailment) |
| API | Request IDs, timeouts, limits, error shape, SSE | Auth, rate limiting, async ingestion jobs |
| Observability | JSON logs, per-stage latency, tokens, cost estimate | Metrics, dashboards, tracing, alerts |
| Eval | 20-question harness, scoring tests | A real run; larger set from real filings; LLM-judged correctness |
| Deploy | Docker, compose, non-root, health check | Cloud deployment (written, not run); external vector store |

## 3. Real difficulties hit while building

From git history and today's fixes, in the order they happened:

1. **Bugs from building stage by stage** (Sep 2026): `load_pdf` and `_merge_chunks` built their lists but had no `return`; `_split_by_paragraphs` raised on empty paragraphs it then filtered anyway; `import fitz` triggered PyMuPDF deprecation warnings. Each failed visibly on the first smoke test.
2. **Features that looked done but weren't.** An audit (`ad64610`) found `chunk_pages(overlap=...)` accepted and ignored, and `embed_chunks(batch_size=100)` still making one API call per chunk; `batch_size` only controlled a progress print. Both fixed in `aacd324`, with tests that assert call counts rather than return values.
3. **Hybrid search broke the abstention signal.** RRF scores are about 0.03 at most, so thresholding on `similarity` would have abstained on everything in hybrid mode. Fix: keep the raw cosine in `vector_similarity` through fusion; tested in `test_fusion_preserves_best_vector_similarity_for_abstention`.
4. **Re-ingest left stale chunks.** Upsert by `source::chunk_index` overwrote matching ids, but if the new version had fewer chunks, the old extras stayed searchable. Fix (`2ac33a1`): embed first, then `delete_source`, then store.
5. **PDF extraction drops blank lines** (`fde96d7`). Running the eval corpus through PyMuPDF showed paragraphs joined by single newlines, so a whole page was one "paragraph" and `CHUNK_MAX_SIZE` was silently exceeded. Fix: `_split_oversized()` splits at lines, then sentences, then a hard cut.
6. **`load_document("x.pdf")` never worked with a path** (`e7dc2ef`). A `str` has no `.name`. The API passes bytes plus a filename, so it never showed up until the eval runner used paths.
7. **Chroma's batch limit** (`34983db`). Reproduced: chromadb 1.5.9 rejects a single upsert of 5462 items. Combined with fix 4, a re-upload of a huge document would have deleted the old copy and then failed to store the new one.
8. **XSS through citations** (`66ce7a3`). The UI put raw chunk text into `unsafe_allow_html`. Fixed with `html.escape`, and verified headlessly that `<script>` renders escaped.
9. **Truncation from thinking** (`78cc1f2`). `claude-opus-5` thinks by default, and thinking counts against `max_tokens` (it was 1024). Raised the default to 4096, and `stop_reason == "max_tokens"` is now an error instead of a silently half-finished answer.
10. **Streaming plus context variables.** The SSE body runs in worker threads after the middleware returns, so pipeline code couldn't see the request's trace. Fix: capture the context once and step the generator with `ctx.run(next, gen)`; tested in `test_stream_logs_summary_after_last_event`.

## 4. Edge cases and what the code does

| Case | Behavior | Where |
|---|---|---|
| Empty PDF / whitespace-only file | No chunks → **422** "No extractable text … may be a scanned PDF" | `api._ingest` |
| Scanned PDF (no OCR layer) | Pages kept, text empty. All empty → 422. Some empty → ingested; `empty_pages` returned and the UI warns. **No OCR.** | `load_pdf`, `IngestResponse.empty_pages` |
| Table-heavy page | `get_text()` returns cells as lines; headers detach from numbers. Chunked like prose. **Not handled**: an answer can pair a figure with the wrong row. | `load_pdf` |
| Huge document | Over `MAX_UPLOAD_MB` (25) → **413** before parsing. Under it: embedded in batches of 100, stored in batches ≤5461. May exceed the 60s timeout → **504**, while the thread still finishes in the background. | `api.ingest`, `storage.store_chunks` |
| No relevant chunk | Gate 1 (threshold) — likely ineffective at 0.3. Gate 2 catches it if the model says not found or cites nothing valid → "Not found in the provided documents." | `llm.retrieval_is_weak`, `finalize_answer` |
| Duplicate chunks | Identical text after case and whitespace normalization: embedded once (`embed_chunks`), sent to the model once (`dedupe_chunks`). Near-duplicates are kept. | `embeddings.py`, `prompt_builder.py` |
| Injection text in a document | Excerpts are fenced and labeled untrusted; our tags are escaped inside them; the UI escapes HTML. Whether the model *obeys* the framing is unmeasured: that's what eval q14 and q15 test. | `prompt_builder.py`, `streamlit_app.py` |
| Very long question | Over 2000 chars → **422**. Delimiter tags in the question are escaped. | `schemas.QueryRequest`, `_neutralize` |
| API timeout | Pipeline over `REQUEST_TIMEOUT_SECONDS` → **504**; SDK clients have the same timeout. Mid-stream failure → SSE `error` event (the HTTP status is already 200). | `api._with_timeout`, `query_stream` |
| Model cites a non-existent excerpt | Marker stripped from the text, id reported in `invalid_citation_ids`; if nothing valid remains → abstain. | `citations.validate_citations` |
| Model output cut off | `max_tokens` stop → 502 with "Raise the limit", never a partial answer | `llm._check_stop` |

## 5. Trade-offs and three alternative designs

**Trade-offs made**

- **No framework (LangChain or LlamaIndex).** Every stage is a plain function I can step through and test; the cost is writing loaders and splitters myself. Bugs 5–7 above are exactly the kind a framework would have hidden; I'd rather have found them.
- **Characters, not tokens, for sizing.** No tokenizer dependency or network call; chunk sizes are approximate in tokens.
- **Chroma embedded on local disk.** Zero setup; it's also the reason deployment is hard (stateless containers).
- **Abstain over guess.** For financial figures a wrong number is worse than no number, so an uncited answer is withheld even if it might be right. This will lower "answer correct" on some questions; the eval reports false abstentions separately.
- **Hybrid and rerank off by default.** The vector-only path is the most tested; the flags exist so the eval can compare.

**Alternative 1: long context, no retrieval.** These documents are small. Send whole documents to Claude (1M-token context) and use the API's native citations (`citations: {enabled: true}` on document blocks), which return cited spans with page locations. Gains: no chunking or retrieval misses, citations validated by the API. Costs: input tokens per query grow with corpus size, so this is only viable for a few documents per question.

**Alternative 2: structured extraction plus SQL for figures.** At ingest, have the model extract tables and key figures into rows (metric, period, value, unit, page), validated against a schema. Answer numeric questions with SQL, and fall back to RAG for narrative. Gains: exact numbers, arithmetic and comparisons work. Costs: an extraction step to evaluate, plus schema design.

**Alternative 3: agentic multi-step retrieval** (LangGraph or a tool loop). The model rewrites the query, retrieves, notices missing pieces ("I have revenue but not the prior year"), and retrieves again before answering. Gains: multi-hop questions. Costs: several LLM calls per question (latency and cost), and harder to evaluate and to bound.

## 6. Security and privacy

| Concern | Status |
|---|---|
| **Authentication / access control** | **NOT implemented.** Anyone who can reach the API can ingest, delete any document by filename, and spend API credits. DEPLOYMENT.md deploys with `--no-allow-unauthenticated` as a stopgap. |
| **Tenant isolation** | **NOT implemented.** One shared collection; any user's query can retrieve any user's documents. Same filename from two users overwrites. |
| **Rate limiting / abuse** | **NOT implemented.** Only per-request limits (size, length, `top_k`). |
| **PII** | **NOT implemented.** No detection or redaction. Document text goes to OpenAI (embeddings) and Anthropic (generation), and is stored in plaintext in Chroma and in the embedding cache. |
| **Prompt injection** | **Partially mitigated:** untrusted-data framing, delimiter escaping, citation validation (an injected "cite nothing" leads to abstention), HTML escaping in the UI. **Not proven:** no measured result yet. No classifier for injected content. |
| **Secrets** | Env vars only; `.env` is gitignored and excluded from the Docker build; Secret Manager in the Cloud Run plan. |
| **Data deletion** | `DELETE /documents/{source}` removes vectors. **Not** removed: embedding-cache entries for that text, and the provider-side logs governed by OpenAI and Anthropic retention policies. |
| **Logging** | Logs hold request IDs, sizes and counts, **not** question or document text. Unhandled-error logs include the exception `repr`, which could contain text. |

## 7. Scaling and concurrency: what breaks first

1. **The vector store.** Chroma `PersistentClient` is SQLite and index files in one process. Running more than one Uvicorn worker or container against the same files risks corruption, so the app is capped at one process today. Fix: Chroma server or pgvector (DEPLOYMENT.md).
2. **Hybrid retrieval.** `_bm25_search` loads every chunk (`get_all_chunks`) and rebuilds BM25 on **every** query: O(corpus) time and memory per request. Off by default for this reason; fix with a persisted, incremental index.
3. **Worker threads.** All pipeline work is blocking code run in AnyIO's thread pool, **40 threads** by default (checked: `current_default_thread_limiter().total_tokens == 40`). That's at most 40 concurrent pipeline calls per process; streams hold a thread while generating. Timed-out work keeps its thread.
4. **Provider rate limits.** OpenAI and Anthropic limits are per key; under load, 429s become 502s once the SDK's default retries are exhausted.
5. **Ingest in the request path.** A big upload holds a request and a thread for the whole embed; it can hit the 504 while still writing. Fix: queue ingestion as a background job and return a job ID.
6. **Memory.** Uploads are read fully into memory (capped at 25 MB) and parsed in memory.

## 8. Hard questions about this repo, with honest answers

1. **How do you know the citations are correct?** I know each one points at a chunk that was retrieved and sent (`validate_citations`). I don't know the chunk supports the claim; no entailment check exists. That would be the next addition: a per-sentence support check, or the API's native citations.
2. **What's your accuracy?** Unknown. The harness and 20 questions are built, but it hasn't run against the real APIs. I won't quote a number I haven't measured.
3. **Why 1000 characters per chunk?** It was a default, not a tuned value. It keeps one financial paragraph or table section together. The eval's hit@k is where I'd tune it, together with `top_k`.
4. **Why is the similarity threshold 0.3?** Honestly, it's uncalibrated, and on this scale (`(1 + cos) / 2`) it probably never fires. Gate 2 does the real work today. The eval prints the similarity distribution for answerable vs unanswerable questions so the threshold can be set from data.
5. **What happens when BM25 and vectors disagree?** Reciprocal rank fusion: each list contributes `1 / (60 + rank)`. Ranks, not scores, because cosine and BM25 scales aren't comparable. A chunk both methods rank decently beats one that only one method ranks first.
6. **Your "reranker" — is that real?** No. `_lexical_rerank` counts query-term overlap. It exists so the flag, call site and tests are real; it shouldn't be called a reranker in the ML sense. A cross-encoder would replace it.
7. **What stops a document from hijacking the model?** Fencing in `<document>` tags, an explicit "this is untrusted data" instruction, escaping our own tags inside document text, and validation downstream: if the model follows an injected "cite nothing", it gets withheld. What's missing is measurement; eval q14 and q15 exist for that.
8. **Why stream at all if you replace the text at the end?** Perceived latency: the user sees progress. The trade-off is that the draft can briefly show a citation that later gets stripped. A stricter design would buffer and send only validated text.
9. **What if the model writes "[2024]"?** It's parsed as citation id 2024, which is out of range, so it's reported invalid and removed from the text. A year in brackets would be lost. That's a known false positive of the regex.
10. **How do you handle a 300-page 10-K?** Under 25 MB it's ingested: about one chunk per 1000 characters, embedded 100 per call, stored in batches. It may exceed the 60s request timeout (504) while still finishing in the background. A job queue is the fix.
11. **Tables?** Badly. PyMuPDF text extraction loses table structure. Alternative 2 (structured extraction) is the real answer.
12. **Is ingest idempotent?** Mostly: same filename leads to delete-then-store, and the cache prevents paying twice to embed unchanged text. It's not atomic: a crash between delete and store loses that document until re-uploaded.
13. **Why Chroma?** Embedded, zero setup, metadata filters, cosine distance. Its cost is exactly the deployment problem: it's local-disk state.
14. **How would you go multi-tenant?** Authenticate, then add a `tenant_id` to every chunk's metadata and to every `where` filter in `storage.py`, or a collection per tenant. Namespace the source ids. Today nothing isolates users.
15. **How do you measure cost?** Real token counts from each provider's `usage` field, priced from `pricing.toml` and labeled as an estimate in every log line. I have the mechanism but no real traffic yet.
16. **What dominates cost?** Probably Claude output tokens, because thinking is billed as output at 5× the input price. That's a hypothesis to confirm in the logs, not a measurement.
17. **Why `claude-opus-5`?** It was the configured default. COST.md notes `claude-opus-5-5` is listed cheaper ($4 / $20 vs $5 / $25), and Sonnet and Haiku cheaper still. The model is one env var; the eval should decide.
18. **What does your test suite actually prove?** 121 offline tests: chunking bounds, citation validation, both abstention gates, injection-tag escaping, the API error shape and limits, the SSE event sequence, log contents, and real-Chroma storage including the batch limit. They mock both AI APIs, so they prove the wiring and rules, not answer quality.
19. **Why chars/4 for tokens?** No network call and no new dependency, and it's only used to bound context size. Real counts come back in `usage`. If the budget mattered tightly, I'd use the token-counting endpoint.
20. **What breaks first under load?** The single-process SQLite vector store, then BM25's per-query full scan if hybrid is on, then the 40-thread pool and provider rate limits.
21. **How do you debug a bad answer in production?** Take the request ID from the response or the `X-Request-ID` header and find its log line: stages, `best_vector_similarity`, `chunks_sent`, dropped duplicates and budget, abstain reason, invalid citations. What's not logged is the question and excerpt text (a privacy choice), so reproducing means re-running the query.
22. **Why not LangChain?** I wanted to understand each stage; I built an earlier LangChain notebook version separately. Here, building it by hand is how I found the PDF blank-line, batch-limit and path bugs. I'd adopt a framework where it adds something I'd otherwise build badly, such as agent orchestration.
23. **What would you change first with a week?** Run the eval and calibrate the threshold; move the vector store out of process; add auth and tenant isolation; add an entailment check on citations.
24. **Is the API safe to expose publicly?** No. No authentication, and every query costs money. It's a prototype behind a private endpoint at best.
25. **What did you build vs what did an AI assistant build?** I built stages 1–2 myself and directed the rest with an AI coding assistant. I reviewed the changes, and I can explain every module and trade-off here. The commit history shows the progression, including the bugs listed in section 3.
