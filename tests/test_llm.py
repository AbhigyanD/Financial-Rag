"""Tests for llm.py — both abstention gates and streaming. Claude is mocked;
no API calls are made.
"""

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

import financial_rag.llm as L
from financial_rag.llm import (
    LLMError,
    finalize_answer,
    generate_answer,
    generate_answer_stream,
    retrieval_is_weak,
)
from financial_rag.prompt_builder import ABSTAIN_TEXT, build_prompt
from financial_rag.retrieval import RetrievedChunk


def _chunk(sim=0.8, idx=0, text="Revenue grew 20% YoY.", page=1):
    return RetrievedChunk(
        page_number=page, text=text, chunk_index=idx, source="report.pdf",
        similarity=sim, vector_similarity=sim,
    )


@pytest.fixture
def threshold(monkeypatch):
    def _set(value):
        monkeypatch.setattr(L, "settings", dataclasses.replace(L.settings, similarity_threshold=value))
    return _set


@pytest.fixture
def mock_client():
    with patch("financial_rag.llm._get_client") as get_client:
        client = MagicMock()
        get_client.return_value = client
        yield client


def _response(text, stop_reason="end_turn", tokens_in=500, tokens_out=40):
    block = MagicMock(type="text", text=text)
    return MagicMock(
        content=[block], stop_reason=stop_reason,
        usage=MagicMock(input_tokens=tokens_in, output_tokens=tokens_out),
    )


# --- gate 1: weak retrieval ------------------------------------------------


def test_no_chunks_is_weak():
    assert retrieval_is_weak([])


def test_weak_when_best_score_below_threshold(threshold):
    threshold(0.6)
    assert retrieval_is_weak([_chunk(0.5), _chunk(0.4)])
    assert not retrieval_is_weak([_chunk(0.5), _chunk(0.7)])


def test_weak_check_uses_vector_similarity_not_rrf_score(threshold):
    threshold(0.6)
    hybrid = _chunk(0.9)
    hybrid["similarity"] = 0.03  # an RRF score — must not trigger abstention
    assert not retrieval_is_weak([hybrid])


def test_weak_retrieval_abstains_without_calling_claude(threshold, mock_client):
    threshold(0.6)
    answer = generate_answer("q", [_chunk(0.2)])

    assert answer["abstained"] and answer["abstain_reason"] == "weak_retrieval"
    assert answer["text"] == ABSTAIN_TEXT
    mock_client.messages.create.assert_not_called()


# --- gate 2: after the model answers ---------------------------------------


def test_model_saying_not_found_is_an_abstention():
    prompt = build_prompt("q", [_chunk()])
    answer = finalize_answer(ABSTAIN_TEXT, prompt)
    assert answer["abstained"] and answer["abstain_reason"] == "model_found_no_support"


def test_answer_with_no_valid_citation_is_an_abstention():
    prompt = build_prompt("q", [_chunk()])
    answer = finalize_answer("Revenue grew a lot.", prompt)
    assert answer["abstained"] and answer["abstain_reason"] == "no_valid_citations"


def test_answer_citing_only_a_fake_excerpt_is_an_abstention():
    prompt = build_prompt("q", [_chunk()])
    answer = finalize_answer("Revenue grew 20% [4].", prompt)
    assert answer["abstained"]
    assert answer["invalid_citation_ids"] == [4]


def test_supported_answer_keeps_valid_citations_and_flags_invalid():
    prompt = build_prompt("q", [_chunk(page=7)])
    answer = finalize_answer("Revenue grew 20% [1]. Also X [3].", prompt)

    assert not answer["abstained"]
    assert [c["page_number"] for c in answer["citations"]] == [7]
    assert answer["invalid_citation_ids"] == [3]
    assert "[3]" not in answer["text"]


def test_generate_answer_reports_real_token_usage(mock_client):
    mock_client.messages.create.return_value = _response("Grew 20% [1].", tokens_in=812, tokens_out=9)
    answer = generate_answer("q", [_chunk()])

    assert answer["input_tokens"] == 812
    assert answer["output_tokens"] == 9


def test_truncated_answer_raises_instead_of_returning_half_an_answer(mock_client):
    mock_client.messages.create.return_value = _response("Revenue was $41", stop_reason="max_tokens")
    with pytest.raises(LLMError, match="cut off"):
        generate_answer("q", [_chunk()])


def test_generate_answer_raises_on_refusal(mock_client):
    mock_client.messages.create.return_value = _response("", stop_reason="refusal")
    with pytest.raises(LLMError, match="declined"):
        generate_answer("q", [_chunk()])


# --- streaming --------------------------------------------------------------


def _mock_stream(fragments, stop_reason="end_turn"):
    cm = MagicMock()
    entered = cm.__enter__.return_value
    entered.text_stream = iter(fragments)
    entered.get_final_message.return_value = MagicMock(
        stop_reason=stop_reason, usage=MagicMock(input_tokens=100, output_tokens=5)
    )
    return cm


def test_stream_yields_deltas_then_validated_final(mock_client):
    mock_client.messages.stream.return_value = _mock_stream(["Grew ", "20% [1]", " [9]."])
    events = list(generate_answer_stream("q", [_chunk()]))

    assert [e[0] for e in events] == ["delta", "delta", "delta", "final"]
    final = events[-1][1]
    assert final["invalid_citation_ids"] == [9]
    assert "[9]" not in final["text"]


def test_stream_weak_retrieval_emits_only_final(mock_client):
    events = list(generate_answer_stream("q", []))
    assert events == [("final", events[0][1])]
    assert events[0][1]["abstain_reason"] == "weak_retrieval"
    mock_client.messages.stream.assert_not_called()


def test_stream_raises_on_refusal(mock_client):
    mock_client.messages.stream.return_value = _mock_stream(["partial"], stop_reason="refusal")
    with pytest.raises(LLMError, match="declined"):
        list(generate_answer_stream("q", [_chunk()]))
