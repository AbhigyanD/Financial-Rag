# Financial RAG

Question answering over uploaded financial documents (PDF or TXT) that cites every claim to a file and page, or answers "Not found in the provided documents." A citation is shown only if it points at a chunk that was actually retrieved and sent to the model.

FastAPI backend, Streamlit thin client, Chroma vector store, OpenAI embeddings, Claude for generation. No RAG framework: each stage is a small module with its own tests.

| Doc | What's in it |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Every layer, how citations are produced, known gaps, implemented vs would-add |
| [INTERVIEW_WALKTHROUGH.md](INTERVIEW_WALKTHROUGH.md) | Request trace, difficulties hit, edge cases, trade-offs, security, scaling, hard questions |
| [COST.md](COST.md) | Where the money goes, how it's logged, levers |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Local Docker (verified), Cloud Run plan (not deployed), hosting comparison |
| EVAL_RESULTS.md | Written by `eval/run_eval.py`. **Not generated yet**: needs API keys |

## Run it

```bash
cp .env.example .env        # set OPENAI_API_KEY and ANTHROPIC_API_KEY
```

With Docker:

```bash
docker compose up --build   # API http://localhost:8000 (docs at /docs), UI http://localhost:8501
```

Without Docker, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run uvicorn financial_rag.api:app --reload      # terminal 1
uv run streamlit run streamlit_app.py              # terminal 2
```

## Test and evaluate

```bash
uv run pytest                       # offline: AI APIs are mocked; 121 tests
uv run python eval/run_eval.py      # REAL API calls, costs money; writes EVAL_RESULTS.md
```

## API

| Method | Path | Does |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/ingest` | Upload a PDF/TXT (multipart `file`): load, chunk, embed, store. 413 over `MAX_UPLOAD_MB`; 422 if no text could be extracted |
| `POST` | `/query` | `{"query", "top_k"?, "source"?}` returns `{answer, abstained, abstain_reason, citations[], invalid_citation_ids[], request_id}` |
| `POST` | `/query/stream` | Same, as Server-Sent Events: `delta`* then `final` (the validated response), or `error` |
| `DELETE` | `/documents/{source}` | Remove one document's chunks |
| `GET` | `/documents/count` | Chunks indexed |

Every response carries `X-Request-ID`. Errors look like `{"error": {"status", "message", "request_id"}}`.

## Configuration

All settings are environment variables; `.env.example` lists each one with its default. Main ones:

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_MAX_SIZE` / `CHUNK_OVERLAP` | 1000 / 100 | Characters, not tokens |
| `HYBRID_RETRIEVAL` | false | Add BM25 + reciprocal rank fusion |
| `RERANK` | false | Naive lexical reorder (not a trained reranker) |
| `SIMILARITY_THRESHOLD` | 0.3 | Abstain without calling the LLM below this. **Uncalibrated** |
| `CONTEXT_TOKEN_BUDGET` | 4000 | Estimated tokens of excerpts per prompt |
| `CLAUDE_MODEL` / `CLAUDE_MAX_TOKENS` | claude-opus-5 / 4096 | Generation model and output cap |
| `REQUEST_TIMEOUT_SECONDS` | 60 | Per-request limit (504) and SDK timeout |

## Layout

```
src/financial_rag/
  loaders/document_loader.py  1  PDF/TXT -> pages
  loaders/chunker.py          2  pages -> size-bounded chunks
  embeddings.py, embedding_cache.py   3  batched, cached embeddings
  storage.py                  4  Chroma wrapper
  retrieval.py                5  vector [+ BM25/RRF] [+ rerank]
  prompt_builder.py           6  dedupe, budget, delimiters
  llm.py                      7  generation + abstention
  citations.py                8  citation validation
  api.py, schemas.py, errors.py       9  HTTP layer
  observability.py            JSON logs, latency, tokens, cost estimate
  config.py                   settings from env
streamlit_app.py              10  UI (HTTP client only)
eval/                         questions, fictional corpus, runner, scoring
tests/                        offline test suite
```

## Status

Built and tested offline, and the containers run locally. **Not yet run end to end against the real OpenAI and Anthropic APIs**, so there are no measured accuracy, latency or cost numbers yet; see INTERVIEW_WALKTHROUGH.md for exactly what is and isn't verified.

## AI usage

An AI coding assistant was used to help structure modules, write tests and docs, and find bugs (several are listed with their commits in INTERVIEW_WALKTHROUGH.md §3). Stages 1–2 were written by hand. Changes were reviewed before committing.
