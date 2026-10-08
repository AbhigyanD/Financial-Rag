"""Tests for api.py — request IDs, error shape, input limits, abstention
and streaming wiring. Pipeline functions are patched; no API calls.
"""

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

import financial_rag.api as A
from financial_rag.llm import Answer
from financial_rag.retrieval import RetrievedChunk

client = TestClient(A.app)

CHUNK = RetrievedChunk(
    page_number=4, text="Revenue was $4.2B.", chunk_index=0, source="10k.pdf",
    similarity=0.8, vector_similarity=0.8,
)

ANSWER = Answer(
    text="Revenue was $4.2B [1].",
    citations=[{
        "id": 1, "source": "10k.pdf", "page_number": 4, "chunk_index": 0,
        "similarity": 0.8, "text": "Revenue was $4.2B.",
    }],
    abstained=False, abstain_reason=None, invalid_citation_ids=[],
    input_tokens=300, output_tokens=12,
)


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(A, "retrieve", lambda q, top_k, source: [CHUNK])
    monkeypatch.setattr(A, "generate_answer", lambda q, chunks: ANSWER)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_every_response_carries_a_request_id():
    assert len(client.get("/health").headers["X-Request-ID"]) == 12


def test_safe_client_request_id_is_echoed():
    r = client.get("/health", headers={"X-Request-ID": "trace-abc_1"})
    assert r.headers["X-Request-ID"] == "trace-abc_1"


def test_unsafe_client_request_id_is_replaced():
    r = client.get("/health", headers={"X-Request-ID": "bad id\n<script>"})
    assert r.headers["X-Request-ID"] != "bad id\n<script>"


def test_query_returns_answer_citations_and_request_id(pipeline):
    r = client.post("/query", json={"query": "Revenue?"}, headers={"X-Request-ID": "q1"})
    body = r.json()

    assert r.status_code == 200
    assert body["citations"][0]["page_number"] == 4
    assert body["abstained"] is False
    assert body["request_id"] == "q1"


@pytest.mark.parametrize(
    "payload",
    [{"query": "   "}, {"query": "x" * 2001}, {"query": "ok", "top_k": 0}, {"query": "ok", "top_k": 21}],
)
def test_bad_query_input_gets_422_in_the_standard_error_shape(payload):
    r = client.post("/query", json=payload)
    assert r.status_code == 422
    assert set(r.json()["error"]) == {"status", "message", "request_id"}


def test_retrieval_failure_maps_to_502(monkeypatch):
    def _boom(q, top_k, source):
        raise A.RetrievalError("vector store down")

    monkeypatch.setattr(A, "retrieve", _boom)
    r = client.post("/query", json={"query": "q"})

    assert r.status_code == 502
    assert "vector store down" in r.json()["error"]["message"]


def test_timeout_maps_to_504(pipeline, monkeypatch):
    import time

    monkeypatch.setattr(A, "settings", dataclasses.replace(A.settings, request_timeout_seconds=0.05))
    monkeypatch.setattr(A, "generate_answer", lambda q, c: time.sleep(0.5) or ANSWER)

    r = client.post("/query", json={"query": "q"})
    assert r.status_code == 504


def test_oversized_upload_is_rejected_with_413(monkeypatch):
    monkeypatch.setattr(A, "settings", dataclasses.replace(A.settings, max_upload_mb=0.001))
    r = client.post("/ingest", files={"file": ("big.txt", b"x" * 5000)})
    assert r.status_code == 413


def test_ingest_with_no_extractable_text_is_rejected_with_422():
    r = client.post("/ingest", files={"file": ("empty.txt", b"   \n\n  ")})
    assert r.status_code == 422
    assert "No extractable text" in r.json()["error"]["message"]


def test_stream_emits_deltas_then_final(monkeypatch):
    monkeypatch.setattr(A, "retrieve", lambda q, top_k, source: [CHUNK])
    monkeypatch.setattr(
        A, "generate_answer_stream",
        lambda q, chunks: iter([("delta", "Revenue "), ("delta", "[1]"), ("final", ANSWER)]),
    )

    r = client.post("/query/stream", json={"query": "q"})
    events = [
        (block.split("\n")[0][len("event: "):], json.loads(block.split("\n")[1][len("data: "):]))
        for block in r.text.strip().split("\n\n")
    ]

    assert [e for e, _ in events] == ["delta", "delta", "final"]
    assert events[-1][1]["citations"][0]["source"] == "10k.pdf"
