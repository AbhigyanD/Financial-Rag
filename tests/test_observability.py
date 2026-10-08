"""Tests for observability.py and its wiring in api.py: one JSON summary line
per request, with request ID, stage timings, tokens, and estimated cost.
"""

import dataclasses
import json
import logging

import pytest
from fastapi.testclient import TestClient

import financial_rag.api as A
import financial_rag.observability as O
from financial_rag.llm import Answer
from financial_rag.retrieval import RetrievedChunk


@pytest.fixture
def pricing(tmp_path, monkeypatch):
    path = tmp_path / "pricing.toml"
    path.write_text(
        '[llm."test-llm"]\ninput_per_mtok = 5.0\noutput_per_mtok = 25.0\n'
        '[embedding."test-emb"]\ninput_per_mtok = 0.02\n'
    )
    monkeypatch.setattr(
        O, "settings",
        dataclasses.replace(O.settings, pricing_file=str(path), claude_model="test-llm", embedding_model="test-emb"),
    )
    O._prices.cache_clear()
    yield
    O._prices.cache_clear()


@pytest.fixture
def summaries(caplog):
    """Collect parsed request_complete records emitted during a test."""
    O.logger.propagate = True  # let caplog see it
    caplog.set_level(logging.INFO, logger="financial_rag")
    yield lambda: [
        json.loads(r.getMessage()) for r in caplog.records
        if r.name == "financial_rag" and '"request_complete"' in r.getMessage()
    ]
    O.logger.propagate = False


def test_stage_and_tokens_are_noops_without_a_trace():
    with O.stage("x"):
        O.add_tokens(embedding=5)
    O.note(a=1)  # must not raise


def test_stage_accumulates_into_active_trace():
    trace = O.Trace("r1", "test")
    with O.activate(trace):
        with O.stage("embed"):
            pass
        with O.stage("embed"):
            pass
        O.add_tokens(embedding=10, llm_input=3)
        O.add_tokens(embedding=5)
    assert "embed" in trace.stages_ms
    assert trace.tokens == {"embedding": 15, "llm_input": 3, "llm_output": 0}


def test_cost_estimate_uses_pricing_file(pricing):
    cost = O.estimate_cost_usd({"embedding": 1_000_000, "llm_input": 1_000_000, "llm_output": 100_000})
    assert cost["embedding"] == pytest.approx(0.02)
    assert cost["llm"] == pytest.approx(5.0 + 2.5)
    assert cost["total"] == pytest.approx(7.52)


def test_unpriced_model_reports_unknown_not_zero(pricing, monkeypatch):
    monkeypatch.setattr(O, "settings", dataclasses.replace(O.settings, claude_model="not-in-file"))
    cost = O.estimate_cost_usd({"llm_input": 100, "llm_output": 10, "embedding": 0})
    assert cost["llm"] is None
    assert cost["total"] is None


# --- wiring through the API -------------------------------------------------

CHUNK = RetrievedChunk(page_number=2, text="Net income was $31.2M.", chunk_index=0,
                       source="r.pdf", similarity=0.8, vector_similarity=0.8)
ANSWER = Answer(text="Net income was $31.2M [1].", citations=[{
    "id": 1, "source": "r.pdf", "page_number": 2, "chunk_index": 0, "similarity": 0.8, "text": CHUNK["text"],
}], abstained=False, abstain_reason=None, invalid_citation_ids=[], input_tokens=900, output_tokens=40)


def _fake_retrieve(q, top_k, source):
    O.add_tokens(embedding=7)  # what embed_query would report
    with O.stage("vector_search"):
        pass
    return [CHUNK]


def test_query_logs_one_summary_with_tokens_stages_and_cost(pricing, summaries, monkeypatch):
    monkeypatch.setattr(A, "retrieve", _fake_retrieve)
    monkeypatch.setattr(A, "generate_answer", lambda q, chunks: ANSWER)

    r = TestClient(A.app).post("/query", json={"query": "Net income?"}, headers={"X-Request-ID": "obs-1"})
    assert r.status_code == 200

    [rec] = summaries()
    assert rec["request_id"] == "obs-1"
    assert rec["status"] == 200
    assert rec["tokens"] == {"embedding": 7, "llm_input": 900, "llm_output": 40}
    assert {"vector_search", "generate"} <= set(rec["stages_ms"])
    assert rec["cost_is_estimate"] is True
    assert rec["est_cost_usd"]["llm"] == pytest.approx((900 * 5 + 40 * 25) / 1e6)
    assert rec["abstained"] is False


def test_stream_logs_summary_after_last_event(pricing, summaries, monkeypatch):
    monkeypatch.setattr(A, "retrieve", _fake_retrieve)

    def _stream(q, chunks):
        O.note(chunks_sent=1)  # pipeline code reporting mid-stream must reach the trace
        yield ("delta", "Net income ")
        yield ("final", ANSWER)

    monkeypatch.setattr(A, "generate_answer_stream", _stream)
    r = TestClient(A.app).post("/query/stream", json={"query": "q"}, headers={"X-Request-ID": "obs-2"})
    assert r.status_code == 200

    [rec] = summaries()
    assert rec["request_id"] == "obs-2"
    assert rec["tokens"]["llm_input"] == 900
    assert rec["chunks_sent"] == 1
    assert rec["first_token_ms"] is not None


def test_failed_request_is_logged_with_its_status(summaries, monkeypatch):
    def _boom(q, top_k, source):
        raise A.RetrievalError("down")

    monkeypatch.setattr(A, "retrieve", _boom)
    TestClient(A.app).post("/query", json={"query": "q"})

    [rec] = summaries()
    assert rec["status"] == 502


def test_health_checks_are_not_logged(summaries):
    TestClient(A.app).get("/health")
    assert summaries() == []
