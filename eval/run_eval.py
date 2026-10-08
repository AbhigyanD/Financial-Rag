"""Run the 20-question eval against the real pipeline and write EVAL_RESULTS.md.

Why: every quality number in the docs must come from a run of this script,
so it records the exact command, commit, config, and raw per-question output.

Usage (from the repo root, with OPENAI_API_KEY and ANTHROPIC_API_KEY set):
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --top-k 3 --out EVAL_RESULTS.md

It calls the real OpenAI and Anthropic APIs and spends real money; the cost
estimate for the run is printed and written into the report.

Isolation: uses its own vector store (eval/.chroma_eval) and collection, so
it never touches documents ingested through the app.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"

# Must be set before financial_rag is imported: settings read the env once.
os.environ["PERSIST_DIRECTORY"] = str(EVAL / ".chroma_eval")
os.environ["COLLECTION_NAME"] = "eval"
os.chdir(ROOT)  # pricing.toml and .env resolve relative to the repo root

import pymupdf as fitz  # noqa: E402

from financial_rag.config import settings  # noqa: E402
from financial_rag.embeddings import embed_chunks  # noqa: E402
from financial_rag.llm import generate_answer  # noqa: E402
from financial_rag.loaders.chunker import chunk_pages  # noqa: E402
from financial_rag.loaders.document_loader import load_document  # noqa: E402
from financial_rag.observability import Trace, activate, add_tokens, stage  # noqa: E402
from financial_rag.observability import estimate_cost_usd  # noqa: E402
from financial_rag.retrieval import retrieve  # noqa: E402
from financial_rag.storage import delete_source, store_chunks  # noqa: E402

sys.path.insert(0, str(EVAL))
from scoring import aggregate, score_question  # noqa: E402

PAGE_BREAK = "\n\\f\n"


def build_corpus(out_dir: Path) -> list[Path]:
    """Render the .pages.txt source to a real PDF; copy .txt files as-is."""
    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for src in sorted((EVAL / "corpus").iterdir()):
        if src.name.endswith(".pages.txt"):
            dst = out_dir / src.name.replace(".pages.txt", ".pdf")
            doc = fitz.open()
            for text in src.read_text().split(PAGE_BREAK):
                page = doc.new_page()
                overflow = page.insert_textbox(fitz.Rect(72, 72, 540, 770), text.strip(), fontsize=11)
                if overflow < 0:
                    raise SystemExit(f"{src.name}: a page's text overflows the PDF page")
            dst.write_bytes(doc.tobytes())
            doc.close()
        elif src.suffix == ".txt":
            dst = out_dir / src.name
            dst.write_text(src.read_text())
        else:
            continue
        built.append(dst)
    return built


def ingest(paths: list[Path]) -> tuple[list[dict], dict]:
    trace = Trace("eval-ingest", "eval ingest")
    rows = []
    with activate(trace):
        for path in paths:
            with stage("load"):
                pages = load_document(str(path))
            with stage("chunk"):
                chunks = chunk_pages(pages, settings.chunk_max_size, settings.chunk_overlap)
            with stage("embed"):
                embedded = embed_chunks(chunks)
            with stage("store"):
                delete_source(path.name)
                store_chunks(embedded, source=path.name)
            rows.append({"source": path.name, "pages": len(pages), "chunks": len(chunks)})
    return rows, {"tokens": trace.tokens, "stages_ms": trace.stages_ms}


def run_question(q: dict, top_k: int) -> dict:
    trace = Trace(q["id"], "eval query")
    error = None
    retrieved: list[dict] = []
    answer = {"text": "", "abstained": False, "abstain_reason": None, "citations": [],
              "invalid_citation_ids": [], "input_tokens": 0, "output_tokens": 0}
    t0 = time.perf_counter()
    with activate(trace):
        try:
            retrieved = retrieve(q["question"], top_k=top_k)
            with stage("generate"):
                answer = generate_answer(q["question"], retrieved)
            add_tokens(llm_input=answer["input_tokens"], llm_output=answer["output_tokens"])
        except Exception as e:  # recorded per question; one failure shouldn't end the run
            error = f"{type(e).__name__}: {e}"
    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "error": error,
        "retrieved": [
            {"source": c["source"], "page_number": c["page_number"],
             "vector_similarity": round(c["vector_similarity"], 4)}
            for c in retrieved
        ],
        "best_vector_similarity": round(max((c["vector_similarity"] for c in retrieved), default=0.0), 4),
        "answer": answer["text"],
        "abstain_reason": answer["abstain_reason"],
        "cited": [(c["id"], c["source"], c["page_number"]) for c in answer["citations"]],
        "invalid_citation_ids": answer["invalid_citation_ids"],
        "tokens": trace.tokens,
        "stages_ms": trace.stages_ms,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        **score_question(q, retrieved, answer),
    }


def _pct(n: int, d: int) -> str:
    return "n/a" if d == 0 else f"{n}/{d} ({100 * n / d:.0f}%)"


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return sha + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_report(path: Path, command: str, ingest_rows, ingest_usage, rows, top_k: int) -> None:
    metrics = aggregate(rows)
    total = {"embedding": ingest_usage["tokens"]["embedding"], "llm_input": 0, "llm_output": 0}
    for r in rows:
        for k in total:
            total[k] += r["tokens"].get(k, 0)
    run_cost = estimate_cost_usd(total)
    answered = [r for r in rows if r["type"] != "unanswerable" and not r["error"]]
    unanswerable = [r for r in rows if r["type"] == "unanswerable" and not r["error"]]
    latencies = sorted(r["latency_ms"] for r in rows if not r["error"])

    def sims(group):
        values = sorted(r["best_vector_similarity"] for r in group)
        return f"min {values[0]:.3f} / median {values[len(values) // 2]:.3f} / max {values[-1]:.3f}" if values else "n/a"

    lines = [
        "# Eval results",
        "",
        "Generated by `eval/run_eval.py` — real OpenAI and Anthropic API calls, not mocks.",
        "The corpus is a **fictional** company (Northwind Freight Holdings) written for this eval;",
        "see `eval/corpus/`. Metric definitions are in `eval/scoring.py`.",
        "",
        f"- Command: `{command}`",
        f"- Run at: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Commit: {_git_commit()}",
        f"- LLM: `{settings.claude_model}` (max_tokens {settings.claude_max_tokens}); "
        f"embeddings: `{settings.embedding_model}`",
        f"- top_k {top_k}; hybrid {settings.hybrid_retrieval}; rerank {settings.rerank}; "
        f"similarity threshold {settings.similarity_threshold}; chunk {settings.chunk_max_size} chars, "
        f"overlap {settings.chunk_overlap}; context budget {settings.context_token_budget} est. tokens",
        "",
        "## Corpus ingested",
        "",
        "| Source | Pages | Chunks |",
        "|---|---|---|",
        *[f"| {r['source']} | {r['pages']} | {r['chunks']} |" for r in ingest_rows],
        "",
        "## Metrics",
        "",
        "| Metric | Result |",
        "|---|---|",
        *[f"| {name} | {_pct(*nd)} |" for name, nd in metrics.items()],
        f"| questions that errored | {sum(1 for r in rows if r['error'])}/{len(rows)} |",
        "",
        "## Similarity distribution (for calibrating SIMILARITY_THRESHOLD)",
        "",
        "Best cosine similarity (rescaled to [0, 1]) among each question's retrieved chunks:",
        "",
        f"- Questions with an answer in the corpus: {sims(answered)}",
        f"- Questions with no answer in the corpus: {sims(unanswerable)}",
        "",
        "## Latency and cost",
        "",
        f"- Per-question end-to-end latency (retrieve + generate): "
        + (f"median {latencies[len(latencies) // 2]:.0f} ms, max {latencies[-1]:.0f} ms" if latencies else "n/a"),
        f"- Tokens — ingest embeddings: {ingest_usage['tokens']['embedding']}; "
        f"query embeddings: {sum(r['tokens'].get('embedding', 0) for r in rows)}; "
        f"LLM input: {total['llm_input']}; LLM output: {total['llm_output']} (from API `usage` fields)",
        f"- Estimated cost of this whole run: "
        + (f"${run_cost['total']:.4f}" if run_cost["total"] is not None else "unknown (model missing from pricing.toml)")
        + " — **estimate** from `pricing.toml`, not a bill.",
        "",
        "## Per question",
        "",
        "| id | type | hit@k | correct | abstained | reason | invalid cites | best sim | ms | error |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    fmt = {True: "yes", False: "no", None: "–"}
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['type']} | {fmt[r['retrieval_hit']]} | {fmt[r['answer_correct']]} | "
            f"{fmt[r['abstained']]} | {r['abstain_reason'] or ''} | {len(r['invalid_citation_ids'])} | "
            f"{r['best_vector_similarity']:.3f} | {r['latency_ms']:.0f} | {r['error'] or ''} |"
        )
    lines += ["", "Raw output for every question: `eval/results/latest.jsonl`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="EVAL_RESULTS.md")
    args = parser.parse_args()

    missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Add them to .env (see .env.example).")

    command = "uv run python eval/run_eval.py" + "".join(
        f" {a}" for a in sys.argv[1:]
    )
    questions = [json.loads(line) for line in (EVAL / "questions.jsonl").read_text().splitlines() if line.strip()]

    print(f"Building corpus and ingesting into {settings.persist_directory} ...")
    ingest_rows, ingest_usage = ingest(build_corpus(EVAL / ".build"))

    rows = []
    for q in questions:
        row = run_question(q, args.top_k)
        rows.append(row)
        status = "ERROR" if row["error"] else ("abstained" if row["abstained"] else "answered")
        print(f"{q['id']} [{q['type']}] {status} ({row['latency_ms']:.0f} ms)")

    results_dir = EVAL / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "latest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    write_report(ROOT / args.out, command, ingest_rows, ingest_usage, rows, args.top_k)

    print(f"\nWrote {args.out} and eval/results/latest.jsonl")
    for name, (n, d) in aggregate(rows).items():
        print(f"  {name}: {_pct(n, d)}")


if __name__ == "__main__":
    main()
