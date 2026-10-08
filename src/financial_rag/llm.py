"""Generate a cited answer to a question from retrieved chunks, or abstain.

Why: this is the only module that calls Claude, so it is where the two
abstention gates live — before the call (retrieval too weak to be worth
asking) and after it (the model found no support, or cited nothing real).

Flow: retrieval_is_weak? -> abstain without an API call.
      else build_prompt -> Claude -> finalize_answer:
        model replied with the abstain sentence and no valid citation -> abstain
        no valid citation at all                                      -> abstain
        otherwise -> answer with only validated citations
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Iterator, Literal, TypedDict, Union

import anthropic

from financial_rag.citations import Citation, validate_citations
from financial_rag.config import settings
from financial_rag.observability import note
from financial_rag.prompt_builder import ABSTAIN_TEXT, BuiltPrompt, build_prompt
from financial_rag.retrieval import RetrievedChunk

MODEL = settings.claude_model
MAX_TOKENS = settings.claude_max_tokens

AbstainReason = Literal["weak_retrieval", "model_found_no_support", "no_valid_citations"]


class LLMError(Exception):
    """Raised when answer generation fails."""


class Answer(TypedDict):
    text: str
    citations: list[Citation]  # only citations validated against the sent chunks
    abstained: bool
    abstain_reason: AbstainReason | None
    invalid_citation_ids: list[int]  # [n] markers the model wrote that matched nothing
    input_tokens: int  # from the API's usage field; 0 if no call was made
    output_tokens: int


StreamEvent = Union[tuple[Literal["delta"], str], tuple[Literal["final"], Answer]]


def _get_client() -> anthropic.Anthropic:
    if not settings.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set. Add it to .env or the environment.")
    return anthropic.Anthropic(timeout=settings.request_timeout_seconds)


@contextmanager
def _translate_anthropic_errors() -> Generator[None]:
    try:
        yield
    except anthropic.AuthenticationError as e:
        raise LLMError("Invalid or missing ANTHROPIC_API_KEY.") from e
    except anthropic.NotFoundError as e:
        raise LLMError(f"Invalid model or endpoint: {MODEL}") from e
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after", "unknown")
        raise LLMError(f"Rate limited by Claude API. Retry after {retry_after}s.") from e
    except anthropic.APITimeoutError as e:
        raise LLMError(f"Claude API timed out after {settings.request_timeout_seconds}s.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Claude API error ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Network error connecting to Claude API.") from e


def retrieval_is_weak(chunks: list[RetrievedChunk]) -> bool:
    """True if no chunk clears the similarity threshold.

    Uses vector_similarity (real cosine, rescaled to [0, 1]) rather than
    `similarity`, which holds an RRF score in hybrid mode.
    """
    if not chunks:
        return True
    best = max(c.get("vector_similarity", c["similarity"]) for c in chunks)
    return best < settings.similarity_threshold


def _abstain(reason: AbstainReason, invalid_ids=None, input_tokens=0, output_tokens=0) -> Answer:
    return Answer(
        text=ABSTAIN_TEXT,
        citations=[],
        abstained=True,
        abstain_reason=reason,
        invalid_citation_ids=invalid_ids or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def finalize_answer(
    raw_text: str, prompt: BuiltPrompt, input_tokens: int = 0, output_tokens: int = 0
) -> Answer:
    """Turn the model's raw text into an Answer, applying the post-call gates."""
    check = validate_citations(raw_text, prompt.chunks)
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}

    if not check.citations:
        said_not_found = ABSTAIN_TEXT.lower().rstrip(".") in raw_text.lower()
        reason = "model_found_no_support" if said_not_found else "no_valid_citations"
        return _abstain(reason, check.invalid_ids, **usage)

    return Answer(
        text=check.text,
        citations=check.citations,
        abstained=False,
        abstain_reason=None,
        invalid_citation_ids=check.invalid_ids,
        **usage,
    )


def _check_stop(stop_reason: str | None) -> None:
    if stop_reason == "refusal":
        raise LLMError("Claude declined to answer this query.")
    if stop_reason == "max_tokens":
        # A truncated answer can end mid-figure; better no answer than half of one.
        raise LLMError(
            f"Answer was cut off at CLAUDE_MAX_TOKENS={MAX_TOKENS}. Raise the limit and retry."
        )


def _request(prompt: BuiltPrompt) -> dict:
    note(
        chunks_sent=len(prompt.chunks),
        dropped_duplicates=prompt.dropped_duplicates,
        dropped_for_budget=prompt.dropped_for_budget,
        est_context_tokens=prompt.estimated_context_tokens,
    )
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": prompt.system,
        "messages": [{"role": "user", "content": prompt.user}],
    }


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> Answer:
    if retrieval_is_weak(chunks):
        return _abstain("weak_retrieval")

    prompt = build_prompt(query, chunks)
    client = _get_client()
    with _translate_anthropic_errors():
        response = client.messages.create(**_request(prompt))

    _check_stop(response.stop_reason)
    text = "".join(block.text for block in response.content if block.type == "text")
    return finalize_answer(
        text, prompt, response.usage.input_tokens, response.usage.output_tokens
    )


def generate_answer_stream(query: str, chunks: list[RetrievedChunk]) -> Iterator[StreamEvent]:
    """Yield ("delta", text) fragments as they arrive, then one ("final", Answer).

    Deltas are raw model output; the final Answer is the validated version
    (invalid citations stripped, or replaced by an abstention). A client
    should replace what it rendered from deltas with final["text"].
    """
    if retrieval_is_weak(chunks):
        yield ("final", _abstain("weak_retrieval"))
        return

    prompt = build_prompt(query, chunks)
    client = _get_client()
    parts: list[str] = []
    with _translate_anthropic_errors():
        with client.messages.stream(**_request(prompt)) as stream:
            for text in stream.text_stream:
                parts.append(text)
                yield ("delta", text)
            final_message = stream.get_final_message()

    _check_stop(final_message.stop_reason)

    yield (
        "final",
        finalize_answer(
            "".join(parts),
            prompt,
            final_message.usage.input_tokens,
            final_message.usage.output_tokens,
        ),
    )
