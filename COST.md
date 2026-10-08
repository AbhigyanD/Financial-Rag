# Cost

Where the money goes, how it's measured, and which levers exist. Every price is an **estimate** read from `pricing.toml`; nothing here is a bill.

> **Status:** no real token counts exist yet. The code that records them is built and tested (`observability.py`), but no API keys were available while building this, so no real request has run. The two numbers below that look like measurements are character-count estimates (chars ÷ 4) from a script run offline, and are labeled as such. Real numbers come from the first keyed run of `eval/run_eval.py`. Copy them into the "Measured" section then.

## Where the money goes

| When | What is paid for | Code | Price (pricing.toml) |
|---|---|---|---|
| Ingest, once per **new** chunk | Embedding tokens | `embeddings.embed_chunks` | `text-embedding-3-small`: $0.02 / 1M tokens |
| Every query | Query embedding (skipped if the same question was asked before; cached) | `embeddings.embed_text` | same |
| Every query that passes gate 1 | LLM input: system prompt + excerpts + question | `llm.generate_answer` | `claude-opus-5`: $5 / 1M input tokens |
| Every query that passes gate 1 | LLM output: answer **plus thinking tokens** | same | `claude-opus-5`: $25 / 1M output tokens |
| Queries that fail gate 1 (weak retrieval) | Query embedding only; Claude is never called | `llm.retrieval_is_weak` | — |

Cost per query ≈ `input_tokens × $5/1M + output_tokens × $25/1M` + a negligible query embedding. Ingest cost ≈ `new chunk tokens × $0.02/1M`, paid once per unique chunk text.

**What will probably dominate:** LLM output. Output tokens cost 5× input, and `claude-opus-5` thinks by default; thinking tokens are billed as output. That is a reason to measure, not a measured fact.

## How it's measured

Every request logs one line like this (an example from running the Docker stack without keys, so all token counts are 0):

```json
{"event": "request_complete", "request_id": "docker-test-1", "route": "POST /query", "status": 502,
 "total_ms": 7.0, "stages_ms": {"embed_query": 4.2},
 "tokens": {"embedding": 0, "llm_input": 0, "llm_output": 0},
 "est_cost_usd": {"embedding": 0.0, "llm": 0.0, "total": 0.0}, "cost_is_estimate": true,
 "llm_model": "claude-opus-5", "embedding_model": "text-embedding-3-small", "top_k": 5, "query_chars": 17}
```

The token counts come from the providers' `usage` fields, not from estimates. If a model is missing from `pricing.toml`, its cost is logged as `null` (unknown) rather than `0`.

## Offline size estimates (not API counts)

Measured by running `prompt_builder.build_prompt` over the eval corpus (`eval/corpus/`) with chars ÷ 4 as the token estimate:

| Quantity | Value | How measured |
|---|---|---|
| Eval corpus | 7 chunks, 3,029 chars (≈757 est. tokens to embed) | chunker over `eval/corpus`, sum of chunk lengths |
| One top-5 prompt over that corpus | 765 + 2,481 chars (≈811 est. input tokens) | `build_prompt(question, 5 chunks)`, system + user length |

These say nothing about output or thinking tokens, which is the part most likely to dominate.

## Measured (fill in from a real run)

After `uv run python eval/run_eval.py`, EVAL_RESULTS.md's "Latency and cost" section has real token totals for ingest, 20 query embeddings, LLM input and LLM output, plus the run's estimated total. Per-query numbers are in `eval/results/latest.jsonl`.

## Levers, in the order to try them

| Lever | What changes | Where | Trade-off |
|---|---|---|---|
| Abstain early (implemented) | No LLM call when retrieval is weak | `SIMILARITY_THRESHOLD` | Needs calibration. At 0.3 it almost never fires (see ARCHITECTURE.md). |
| Embedding cache (implemented) | Unchanged chunks and repeat queries cost nothing | `embedding_cache.py` | Disk space; cache keyed on exact text |
| Batching (implemented) | 100 texts per embedding call | `embed_chunks(batch_size=100)` | Fewer calls, same tokens; it saves latency and rate limit, not money |
| Fewer chunks | Smaller prompt | `top_k`, `CONTEXT_TOKEN_BUDGET` | Lower recall; check retrieval hit@k in the eval |
| Smaller chunks | Less irrelevant text per excerpt | `CHUNK_MAX_SIZE` | More chunks to embed; context can fragment |
| Lower effort | Fewer thinking tokens | add `output_config={"effort": "low"}` in `llm._request` (**not implemented**) | Possibly weaker answers on hard questions; measure with the eval |
| Cheaper or newer model | Lower per-token price | `CLAUDE_MODEL` | `claude-opus-5-5` is listed at $4 / $20 (vs $5 / $25). Sonnet and Haiku are cheaper still but need the eval rerun |
| Prompt caching | System prompt billed at a discount on repeats | `cache_control` (**not implemented**) | The system prompt is ~191 est. tokens, likely below the minimum cacheable prefix, so probably no gain here |
| Batch API | 50% off for non-interactive work | **not implemented** | Only for offline jobs like the eval, not live queries |
