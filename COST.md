# Cost

Where the money goes, how it's measured, and which levers exist. Prices come from `pricing.toml` and are **estimates**, never a bill.

## Default setup: no per-token bill

| Stage | Default provider | Cost |
|---|---|---|
| Embeddings (ingest and every query) | all-MiniLM-L6-v2, **local** CPU via fastembed | $0 per token. Costs CPU time and a one-time ~90 MB model download (baked into the Docker image). |
| Answers | Groq `llama-3.3-70b-versatile` | $0 on Groq's free tier, which is **rate-limited**. Paid-tier prices couldn't be verified (Groq's pricing page showed none when fetched), so `pricing.toml` has no Groq entry and logs record LLM cost as `null` (unknown), not $0. |

So the real limits on the default setup are throughput and latency, not money: Groq's free-tier rate limits (a 429 becomes a clear "Rate limited by Groq. Retry after N seconds." error), and local embedding speed on the host CPU.

## If you switch to paid providers

| Setting | What is paid for | Price in pricing.toml |
|---|---|---|
| `EMBEDDING_PROVIDER=openai` | Embedding tokens at ingest (once per **new** chunk; cached after) and per query (cached for repeated questions) | `text-embedding-3-small`: $0.02 / 1M tokens |
| `LLM_PROVIDER=anthropic`, `CLAUDE_MODEL=claude-opus-5` | Input: system prompt + excerpts + question. Output: answer **plus thinking tokens** (thinking is on by default for this model and billed as output) | $5 / 1M input, $25 / 1M output |
| `CLAUDE_MODEL=claude-opus-5-5` | Same | $4 / $20 |

Cost per query with a paid LLM ≈ `input_tokens × input price + output_tokens × output price`. Queries that fail gate 1 (weak retrieval) never call the LLM.

## How it's measured

Every request logs one JSON line with real token counts from the provider's `usage` field (Groq reports streaming usage under `x_groq.usage`; the code reads it). Example from the Docker run, no key set, so no tokens:

```json
{"event": "request_complete", "request_id": "docker-test-1", "route": "POST /query", "status": 502,
 "total_ms": 7.0, "stages_ms": {"embed_query": 4.2},
 "tokens": {"embedding": 0, "llm_input": 0, "llm_output": 0},
 "est_cost_usd": {"embedding": 0.0, "llm": 0.0, "total": 0.0}, "cost_is_estimate": true, ...}
```

That example predates the switch to Groq. With Groq as the model, `est_cost_usd.llm` is `null`, because Groq has no entry in `pricing.toml`. Local embeddings report no token count (there's no API to bill), so `tokens.embedding` stays 0.

## Offline size estimates (not API counts)

From running `prompt_builder.build_prompt` over the eval corpus, with characters ÷ 4 as the token estimate:

| Quantity | Value | How measured |
|---|---|---|
| Eval corpus | 7 chunks, 3,029 chars (≈757 est. tokens) | chunker over `eval/corpus`, before header stripping |
| One top-5 prompt | 765 + 2,481 chars (≈811 est. input tokens) | `build_prompt(question, 5 chunks)` |

Real per-query token counts come from the first `eval/run_eval.py` run with a Groq key (EVAL_RESULTS.md, "Latency and cost").

## Levers

| Lever | Status | Effect |
|---|---|---|
| Abstain early (`SIMILARITY_THRESHOLD=0.62`) | Implemented, measured | Off-topic questions (best similarity 0.506–0.560 in the measurement) never reach the LLM |
| Local embeddings | Implemented, default | No per-token embedding cost at all |
| Embedding cache | Implemented | Unchanged chunks and repeat questions are never re-embedded |
| Batching (100 texts per call) | Implemented | Fewer calls, same tokens: saves latency and rate limit, not money |
| Fewer or smaller chunks (`top_k`, `CONTEXT_TOKEN_BUDGET`, `CHUNK_MAX_SIZE`) | Configurable | Smaller prompts; check retrieval hit@k in the eval when changing |
| Smaller model (`GROQ_MODEL=llama-3.1-8b-instant`) | Configurable | Faster and higher free-tier limits; citation-following must be re-checked with the eval |
| Lower effort (Claude only) | **Not implemented** | Fewer thinking tokens on `claude-opus-5` |
| Prompt caching or Batch API | **Not implemented** | The system prompt is ~191 est. tokens, likely too short to cache; Batch API only suits offline jobs like the eval |
