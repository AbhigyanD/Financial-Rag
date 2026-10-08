"""Generate a cited answer to a question from retrieved chunks, or abstain.

Why: the only module that calls an LLM, so it is where the two abstention
gates live — before the call (retrieval too weak to be worth asking) and
after it (the model found no support, or cited nothing real).

Providers, chosen by LLM_PROVIDER (see config.py):
- "groq" (default): Groq's OpenAI-compatible API, via the `openai` SDK
  pointed at Groq's base URL. Model: GROQ_MODEL.
- "anthropic": Claude via the `anthropic` SDK. Model: CLAUDE_MODEL.
Both are normalized to the same (text, stop reason, token counts) shape,
so everything after the API call is provider-independent.

Flow: retrieval_is_weak? -> abstain without an API call.
      else build_prompt -> LLM -> finalize_answer:
        model replied with the abstain sentence and no valid citation -> abstain
        no valid citation at all                                      -> abstain
        otherwise -> answer with only validated citations
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Generator, Iterator, Literal, TypedDict, Union

import anthropic
import openai

from financial_rag.citations import Citation, validate_citations
from financial_rag.config import settings
from financial_rag.observability import note
from financial_rag.prompt_builder import ABSTAIN_TEXT, BuiltPrompt, build_prompt
from financial_rag.retrieval import RetrievedChunk

PROVIDER = settings.llm_provider
MODEL = settings.llm_model
MAX_TOKENS = settings.llm_max_tokens
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

AbstainReason = Literal["weak_retrieval", "model_found_no_support", "no_valid_citations"]
Stop = Literal["end", "max_tokens", "refusal"]


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


@dataclass(frozen=True)
class _Completion:
    text: str
    stop: Stop
    input_tokens: int
    output_tokens: int


# --- provider: Anthropic ------------------------------------------------------


def _anthropic_client() -> anthropic.Anthropic:
    if not settings.anthropic_api_key:
        raise LLMError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(timeout=settings.request_timeout_seconds)


_ANTHROPIC_STOP: dict[str | None, Stop] = {"max_tokens": "max_tokens", "refusal": "refusal"}


def _anthropic_args(prompt: BuiltPrompt) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": prompt.system,
        "messages": [{"role": "user", "content": prompt.user}],
    }


def _anthropic_complete(prompt: BuiltPrompt) -> _Completion:
    response = _anthropic_client().messages.create(**_anthropic_args(prompt))
    return _Completion(
        text="".join(b.text for b in response.content if b.type == "text"),
        stop=_ANTHROPIC_STOP.get(response.stop_reason, "end"),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def _anthropic_stream(prompt: BuiltPrompt) -> Iterator[str | _Completion]:
    with _anthropic_client().messages.stream(**_anthropic_args(prompt)) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    yield _Completion("", _ANTHROPIC_STOP.get(final.stop_reason, "end"),
                      final.usage.input_tokens, final.usage.output_tokens)


# --- provider: Groq (OpenAI-compatible) ---------------------------------------


@lru_cache(maxsize=1)
def _groq_client() -> openai.OpenAI:
    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set. Add it to .env (keys start with gsk_).")
    return openai.OpenAI(
        api_key=settings.groq_api_key, base_url=GROQ_BASE_URL, timeout=settings.request_timeout_seconds
    )


_OPENAI_STOP: dict[str | None, Stop] = {"length": "max_tokens", "content_filter": "refusal"}


def _groq_args(prompt: BuiltPrompt) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,  # extraction from documents, not creative writing
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
    }


def _groq_complete(prompt: BuiltPrompt) -> _Completion:
    response = _groq_client().chat.completions.create(**_groq_args(prompt))
    choice = response.choices[0]
    usage = response.usage
    return _Completion(
        text=choice.message.content or "",
        stop=_OPENAI_STOP.get(choice.finish_reason, "end"),
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


def _groq_usage(chunk) -> tuple[int, int] | None:
    """Groq reports streaming usage on the last chunk under `x_groq.usage`;
    the OpenAI-standard `chunk.usage` is checked too."""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        usage = ((getattr(chunk, "model_extra", None) or {}).get("x_groq") or {}).get("usage")
    if usage is None:
        return None
    get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, 0)
    return get("prompt_tokens") or 0, get("completion_tokens") or 0


def _groq_stream(prompt: BuiltPrompt) -> Iterator[str | _Completion]:
    stop: Stop = "end"
    tokens = (0, 0)
    for chunk in _groq_client().chat.completions.create(**_groq_args(prompt), stream=True):
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                yield choice.delta.content
            if choice.finish_reason:
                stop = _OPENAI_STOP.get(choice.finish_reason, "end")
        tokens = _groq_usage(chunk) or tokens
    yield _Completion("", stop, *tokens)


# --- provider-independent ------------------------------------------------------


@contextmanager
def _translate_errors() -> Generator[None]:
    """Map either SDK's typed errors to LLMError, most specific first.
    (openai and anthropic share class names but not classes.)"""
    name = "Groq" if PROVIDER == "groq" else "Claude"
    try:
        yield
    except (openai.AuthenticationError, anthropic.AuthenticationError) as e:
        raise LLMError(f"{name} rejected the API key. Check it in .env.") from e
    except (openai.NotFoundError, anthropic.NotFoundError) as e:
        setting = "GROQ_MODEL" if PROVIDER == "groq" else "CLAUDE_MODEL"
        raise LLMError(
            f"{name} doesn't offer the model '{MODEL}' to this API key. "
            f"Set {setting} in .env to a model your account has, then restart."
        ) from e
    except (openai.RateLimitError, anthropic.RateLimitError) as e:
        retry_after = e.response.headers.get("retry-after", "a few")
        raise LLMError(f"Rate limited by {name}. Retry after {retry_after} seconds.") from e
    except (openai.APITimeoutError, anthropic.APITimeoutError) as e:
        raise LLMError(f"{name} timed out after {settings.request_timeout_seconds:.0f}s.") from e
    except (openai.APIStatusError, anthropic.APIStatusError) as e:
        raise LLMError(f"{name} API error ({e.status_code}): {e.message}") from e
    except (openai.APIConnectionError, anthropic.APIConnectionError) as e:
        raise LLMError(f"Network error connecting to {name}.") from e


def _complete(prompt: BuiltPrompt) -> _Completion:
    return _anthropic_complete(prompt) if PROVIDER == "anthropic" else _groq_complete(prompt)


def _stream(prompt: BuiltPrompt) -> Iterator[str | _Completion]:
    return _anthropic_stream(prompt) if PROVIDER == "anthropic" else _groq_stream(prompt)


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


def _check_stop(stop: Stop) -> None:
    if stop == "refusal":
        raise LLMError("The model declined to answer this query.")
    if stop == "max_tokens":
        # A truncated answer can end mid-figure; better no answer than half of one.
        raise LLMError(f"Answer was cut off at LLM_MAX_TOKENS={MAX_TOKENS}. Raise the limit and retry.")


def _note_prompt(prompt: BuiltPrompt) -> None:
    note(
        llm_provider=PROVIDER,
        chunks_sent=len(prompt.chunks),
        dropped_duplicates=prompt.dropped_duplicates,
        dropped_for_budget=prompt.dropped_for_budget,
        est_context_tokens=prompt.estimated_context_tokens,
    )


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> Answer:
    if retrieval_is_weak(chunks):
        return _abstain("weak_retrieval")

    prompt = build_prompt(query, chunks)
    _note_prompt(prompt)
    with _translate_errors():
        result = _complete(prompt)

    _check_stop(result.stop)
    return finalize_answer(result.text, prompt, result.input_tokens, result.output_tokens)


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
    _note_prompt(prompt)
    parts: list[str] = []
    result: _Completion | None = None
    with _translate_errors():
        for item in _stream(prompt):
            if isinstance(item, _Completion):
                result = item
            else:
                parts.append(item)
                yield ("delta", item)

    if result is None:
        raise LLMError("The stream ended without a final message.")
    _check_stop(result.stop)
    yield ("final", finalize_answer("".join(parts), prompt, result.input_tokens, result.output_tokens))
