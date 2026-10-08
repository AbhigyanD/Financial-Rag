"""Tests for llm.py — both abstention gates, both providers, streaming.
Every LLM client is faked; no API calls are made.
"""

import dataclasses
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

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


# --- fake clients ---------------------------------------------------------------


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr(L, "PROVIDER", "groq")
    client = MagicMock()
    monkeypatch.setattr(L, "_groq_client", lambda: client)
    return client


@pytest.fixture
def claude(monkeypatch):
    monkeypatch.setattr(L, "PROVIDER", "anthropic")
    client = MagicMock()
    monkeypatch.setattr(L, "_anthropic_client", lambda: client)
    return client


def _groq_response(text, finish="stop", tokens_in=500, tokens_out=40):
    return NS(
        choices=[NS(message=NS(content=text), finish_reason=finish)],
        usage=NS(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


def _groq_chunks(fragments, finish="stop", usage=(100, 5)):
    """Shape of Groq's OpenAI-compatible stream: content deltas, then a last
    chunk carrying finish_reason and usage under x_groq."""
    chunks = [NS(choices=[NS(delta=NS(content=f), finish_reason=None)], usage=None, model_extra={})
              for f in fragments]
    chunks.append(NS(
        choices=[NS(delta=NS(content=None), finish_reason=finish)], usage=None,
        model_extra={"x_groq": {"usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}},
    ))
    return iter(chunks)


def _claude_response(text, stop_reason="end_turn", tokens_in=500, tokens_out=40):
    return MagicMock(
        content=[MagicMock(type="text", text=text)], stop_reason=stop_reason,
        usage=MagicMock(input_tokens=tokens_in, output_tokens=tokens_out),
    )


def _claude_stream(fragments, stop_reason="end_turn"):
    cm = MagicMock()
    entered = cm.__enter__.return_value
    entered.text_stream = iter(fragments)
    entered.get_final_message.return_value = MagicMock(
        stop_reason=stop_reason, usage=MagicMock(input_tokens=100, output_tokens=5)
    )
    return cm


# --- gate 1: weak retrieval ----------------------------------------------------


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


def test_weak_retrieval_abstains_without_calling_the_llm(threshold, groq):
    threshold(0.6)
    answer = generate_answer("q", [_chunk(0.2)])

    assert answer["abstained"] and answer["abstain_reason"] == "weak_retrieval"
    assert answer["text"] == ABSTAIN_TEXT
    groq.chat.completions.create.assert_not_called()


# --- gate 2: after the model answers (provider-independent) ---------------------


def test_model_saying_not_found_is_an_abstention():
    answer = finalize_answer(ABSTAIN_TEXT, build_prompt("q", [_chunk()]))
    assert answer["abstained"] and answer["abstain_reason"] == "model_found_no_support"


def test_answer_with_no_valid_citation_is_an_abstention():
    answer = finalize_answer("Revenue grew a lot.", build_prompt("q", [_chunk()]))
    assert answer["abstained"] and answer["abstain_reason"] == "no_valid_citations"


def test_answer_citing_only_a_fake_excerpt_is_an_abstention():
    answer = finalize_answer("Revenue grew 20% [4].", build_prompt("q", [_chunk()]))
    assert answer["abstained"]
    assert answer["invalid_citation_ids"] == [4]


def test_supported_answer_keeps_valid_citations_and_flags_invalid():
    answer = finalize_answer("Revenue grew 20% [1]. Also X [3].", build_prompt("q", [_chunk(page=7)]))

    assert not answer["abstained"]
    assert [c["page_number"] for c in answer["citations"]] == [7]
    assert answer["invalid_citation_ids"] == [3]
    assert "[3]" not in answer["text"]


# --- Groq ------------------------------------------------------------------------


def test_groq_answer_and_token_usage(groq):
    groq.chat.completions.create.return_value = _groq_response("Grew 20% [1].", tokens_in=812, tokens_out=9)
    answer = generate_answer("q", [_chunk()])

    assert not answer["abstained"]
    assert (answer["input_tokens"], answer["output_tokens"]) == (812, 9)


def test_groq_request_sends_system_prompt_and_fenced_documents(groq):
    groq.chat.completions.create.return_value = _groq_response("Grew 20% [1].")
    generate_answer("q", [_chunk()])

    messages = groq.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system" and "untrusted DATA" in messages[0]["content"]
    assert messages[1]["role"] == "user" and "<documents>" in messages[1]["content"]


def test_groq_truncation_raises(groq):
    groq.chat.completions.create.return_value = _groq_response("Revenue was $41", finish="length")
    with pytest.raises(LLMError, match="cut off"):
        generate_answer("q", [_chunk()])


def test_groq_stream_reads_usage_from_x_groq(groq):
    groq.chat.completions.create.return_value = _groq_chunks(["Grew ", "20% [1]", " [9]."], usage=(321, 7))
    events = list(generate_answer_stream("q", [_chunk()]))

    assert [e[0] for e in events] == ["delta", "delta", "delta", "final"]
    final = events[-1][1]
    assert final["invalid_citation_ids"] == [9]
    assert "[9]" not in final["text"]
    assert (final["input_tokens"], final["output_tokens"]) == (321, 7)


def test_groq_stream_truncation_raises_after_deltas(groq):
    groq.chat.completions.create.return_value = _groq_chunks(["partial"], finish="length")
    with pytest.raises(LLMError, match="cut off"):
        list(generate_answer_stream("q", [_chunk()]))


def test_missing_groq_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(L, "PROVIDER", "groq")
    monkeypatch.setattr(L, "settings", dataclasses.replace(L.settings, groq_api_key=None))
    L._groq_client.cache_clear()
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        generate_answer("q", [_chunk()])
    L._groq_client.cache_clear()


# --- Anthropic ---------------------------------------------------------------------


def test_claude_answer_and_token_usage(claude):
    claude.messages.create.return_value = _claude_response("Grew 20% [1].", tokens_in=812, tokens_out=9)
    answer = generate_answer("q", [_chunk()])
    assert (answer["input_tokens"], answer["output_tokens"]) == (812, 9)


def test_claude_truncation_raises(claude):
    claude.messages.create.return_value = _claude_response("Revenue was $41", stop_reason="max_tokens")
    with pytest.raises(LLMError, match="cut off"):
        generate_answer("q", [_chunk()])


def test_claude_refusal_raises(claude):
    claude.messages.create.return_value = _claude_response("", stop_reason="refusal")
    with pytest.raises(LLMError, match="declined"):
        generate_answer("q", [_chunk()])


def test_claude_stream_yields_deltas_then_validated_final(claude):
    claude.messages.stream.return_value = _claude_stream(["Grew ", "20% [1]"])
    events = list(generate_answer_stream("q", [_chunk()]))
    assert [e[0] for e in events] == ["delta", "delta", "final"]
    assert events[-1][1]["citations"][0]["source"] == "report.pdf"


def test_stream_weak_retrieval_emits_only_final(groq):
    events = list(generate_answer_stream("q", []))
    assert len(events) == 1 and events[0][0] == "final"
    assert events[0][1]["abstain_reason"] == "weak_retrieval"
    groq.chat.completions.create.assert_not_called()


def test_unavailable_model_error_says_which_setting_to_change(groq):
    import httpx2 as httpx

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    groq.chat.completions.create.side_effect = L.openai.NotFoundError(
        "model_not_found", response=httpx.Response(404, request=request), body=None
    )
    with pytest.raises(LLMError, match="GROQ_MODEL"):
        generate_answer("q", [_chunk()])
