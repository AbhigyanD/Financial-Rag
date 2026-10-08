# Financial RAG

Question answering over uploaded financial documents (PDF or TXT) that cites every claim to a file and page, or answers "Not found in the provided documents." A citation is shown only if it points at a chunk that was actually retrieved and sent to the model.

FastAPI backend with a plain HTML/JS demo page, Chroma vector store, local embeddings (all-MiniLM-L6-v2 via fastembed), and Groq (`openai/gpt-oss-120b`) for answers. Claude and OpenAI embeddings remain available behind config flags. No RAG framework: each stage is a small module with its own tests.

| Doc | What's in it |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Every layer, how citations are produced, known gaps, implemented vs would-add |
| [INTERVIEW_WALKTHROUGH.md](INTERVIEW_WALKTHROUGH.md) | Request trace, difficulties hit, edge cases, trade-offs, security, scaling, hard questions |
| [COST.md](COST.md) | Where the money goes, how it's logged, levers |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Local Docker (verified), Cloud Run plan (not deployed), hosting comparison |
| [EVAL_RESULTS.md](EVAL_RESULTS.md) | Real 20-question run on Groq, a two-model comparison, and what the numbers do and don't show |

## Run the demo

```bash
cp .env.example .env        # paste your GROQ_API_KEY (free at console.groq.com)
```

With Docker:

```bash
docker compose up --build   # open http://localhost:8000
```

Without Docker, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run uvicorn financial_rag.api:app --reload      # open http://localhost:8000
```

In the page, click **Load the sample filings** (three short documents for a fictional company, two of which contain prompt-injection text), then try the suggested questions. The first ingest downloads the embedding model once (~90 MB) unless you're using the Docker image, which already contains it.

The older Streamlit client still works: `uv run streamlit run streamlit_app.py` (port 8501).

## Test and evaluate

```bash
uv run pytest                       # offline: every LLM client is faked; 134 tests
uv run python eval/run_eval.py --pause 10   # REAL model calls on Groq; writes EVAL_RESULTS.md
```

## API

| Method | Path | Does |
|---|---|---|
| `GET` | `/` | The demo web page |
| `GET` | `/health` | Liveness |
| `GET` | `/info` | Models in use, and whether the LLM key is set (never the key) |
| `POST` | `/ingest` | Upload a PDF/TXT (multipart `file`): load, chunk, embed, store. 413 over `MAX_UPLOAD_MB`; 422 if no text could be extracted |
| `GET` | `/documents` | Stored documents with page and passage counts |
| `POST` | `/query` | `{"query", "top_k"?, "source"?}` returns `{answer, abstained, abstain_reason, citations[], invalid_citation_ids[], request_id}` |
| `POST` | `/query/stream` | Same, as Server-Sent Events: `delta`* then `final` (the validated response), or `error` |
| `DELETE` | `/documents/{source}` | Remove one document's chunks |
| `GET` | `/documents/count` | Chunks indexed |

Every response carries `X-Request-ID`. Errors look like `{"error": {"status", "message", "request_id"}}`.

## Configuration

All settings are environment variables; `.env.example` lists each one with its default. Main ones:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` / `GROQ_MODEL` | groq / openai/gpt-oss-120b | `anthropic` switches to Claude (`CLAUDE_MODEL`) |
| `EMBEDDING_PROVIDER` | local | `openai` switches to text-embedding-3-small |
| `CHUNK_MAX_SIZE` / `CHUNK_OVERLAP` | 1000 / 100 | Characters, not tokens |
| `HYBRID_RETRIEVAL` | false | Add BM25 + reciprocal rank fusion |
| `RERANK` | false | Naive lexical reorder (not a trained reranker) |
| `SIMILARITY_THRESHOLD` | 0.62 | Abstain without calling the LLM below this. Measured for local MiniLM on the eval corpus; re-measure if you change the model |
| `CONTEXT_TOKEN_BUDGET` | 4000 | Estimated tokens of excerpts per prompt |
| `LLM_MAX_TOKENS` | 4096 | Output cap; a cut-off answer is an error, never shown |
| `REQUEST_TIMEOUT_SECONDS` | 60 | Per-request limit (504) and SDK timeout |

## Layout

```
src/financial_rag/
  loaders/document_loader.py  1  PDF/TXT -> pages, running headers stripped
  loaders/chunker.py          2  pages -> size-bounded chunks
  embeddings.py, embedding_cache.py   3  local or OpenAI embeddings, cached, batched
  storage.py                  4  Chroma wrapper
  retrieval.py                5  vector [+ BM25/RRF] [+ rerank]
  prompt_builder.py           6  dedupe, budget, delimiters
  llm.py                      7  Groq or Claude, plus abstention
  citations.py                8  citation validation
  api.py, schemas.py, errors.py       9  HTTP layer
  web/                        10  demo page (HTML/CSS/JS) and sample filings
  observability.py            JSON logs, latency, tokens, cost estimate
  config.py                   settings from env
streamlit_app.py              older UI (HTTP client only)
eval/                         questions, fictional corpus, runner, scoring
tests/                        offline test suite
```

## Status

Run end to end for real on 2026-10-08: the 20-question eval on Groq scored 15/15 correct, 5/5 correct abstentions, 2/2 injections resisted, 15/15 valid citations, 0 errors (median 657 ms per question). The demo page was driven in headless Chrome against the live model. The eval corpus is tiny and fictional, so read EVAL_RESULTS.md's limits section before quoting these numbers.

## AI usage

An AI coding assistant was used to help structure modules, write tests and docs, and find bugs (several are listed with their commits in INTERVIEW_WALKTHROUGH.md §3). Stages 1–2 were written by hand. Changes were reviewed before committing.
