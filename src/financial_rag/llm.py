"""Generate a cited answer to a user's query from retrieved chunks.

This is stage 6 of the pipeline: given a query and the chunks retrieved
for it (stage 5), build a prompt that grounds Claude in only that context
and asks it to answer with page citations, then call the Messages API.

Design notes:
- The prompt instructs Claude to answer ONLY from the provided context and
  to say so explicitly when the context doesn't contain the answer —
  this is what keeps a RAG system from hallucinating financial figures.
- Each chunk is labeled with its source/page in the prompt so Claude can
  cite "[source, page N]" inline, and so a UI can cross-check citations
  against the actual RetrievedChunk objects afterward.
- Model is fixed at module level for consistency; swap here if needed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Iterator, TypedDict

import anthropic

from financial_rag.config import settings
from financial_rag.retrieval import RetrievedChunk

# Configurable via CLAUDE_MODEL / CLAUDE_MAX_TOKENS in .env — see config.py.
MODEL = settings.claude_model
MAX_TOKENS = settings.claude_max_tokens

SYSTEM_PROMPT = """\
You are a financial research assistant. Answer the user's question using \
ONLY the excerpts provided in the context below — do not use outside \
knowledge, and do not guess or extrapolate beyond what the context states.

Rules:
- Cite every claim inline using the format [source, page N], where source \
and page come from the excerpt you're drawing from.
- If the context does not contain enough information to answer the \
question, say so explicitly instead of guessing.
- Be concise and precise — this is financial information; do not round, \
approximate, or restate figures inaccurately.
"""


class LLMError(Exception):
    """Raised when answer generation fails."""


class Answer(TypedDict):
    """The generated answer plus the chunks it was grounded in."""

    text: str
    citations: list[RetrievedChunk]  # the chunks passed as context, for UI cross-checking


def _get_client() -> anthropic.Anthropic:
    """Create the Anthropic client. Resolves ANTHROPIC_API_KEY from the
    environment automatically; raises a clear error if no credentials are
    configured at all.
    """
    try:
        return anthropic.Anthropic()
    except Exception as e:
        raise LLMError(f"Failed to create Anthropic client: {e}") from e


@contextmanager
def _translate_anthropic_errors() -> Generator[None]:
    """Convert the anthropic SDK's typed exceptions into LLMError.

    Shared by generate_answer() and generate_answer_stream() so the same
    most-specific-first exception chain isn't duplicated between the
    non-streaming and streaming call paths.
    """
    try:
        yield
    except anthropic.BadRequestError as e:
        raise LLMError(f"Bad request to Claude API: {e.message}") from e
    except anthropic.AuthenticationError as e:
        raise LLMError("Invalid or missing ANTHROPIC_API_KEY.") from e
    except anthropic.PermissionDeniedError as e:
        raise LLMError("API key lacks permission for this request.") from e
    except anthropic.NotFoundError as e:
        raise LLMError(f"Invalid model or endpoint: {MODEL}") from e
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after", "unknown")
        raise LLMError(f"Rate limited by Claude API. Retry after {retry_after}s.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Claude API error ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Network error connecting to Claude API.") from e


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a labeled context block for the prompt.

    Each chunk is tagged with its source filename and page number so
    Claude can cite it, e.g.:

        [Excerpt 1 — quarterly_report.pdf, page 4]
        Revenue grew 20% year-over-year...
    """
    if not chunks:
        return "(No relevant context was found for this query.)"

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Excerpt {i} — {chunk['source']}, page {chunk['page_number']}]"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> Answer:
    """Generate a cited answer to `query`, grounded in `chunks`.

    Args:
        query: The user's natural-language question.
        chunks: RetrievedChunk list from retrieval.retrieve(), ordered by
            relevance. May be empty — Claude will be told no context was
            found and should say so rather than answer from general
            knowledge.

    Returns:
        An Answer dict with the generated text and the chunks used as
        context (for a UI to cross-reference citations against).

    Raises:
        LLMError: if the API call fails for any reason.
    """
    context = build_context(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {query}"

    client = _get_client()
    with _translate_anthropic_errors():
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

    if response.stop_reason == "refusal":
        raise LLMError("Claude declined to answer this query.")

    answer_text = next(
        (block.text for block in response.content if block.type == "text"), ""
    )

    return Answer(text=answer_text, citations=chunks)


def generate_answer_stream(
    query: str, chunks: list[RetrievedChunk]
) -> Iterator[str]:
    """Like generate_answer(), but yields the answer text incrementally.

    Yields plain text fragments as Claude generates them, for a chat UI
    that wants to render the response token-by-token instead of waiting
    for the full answer. Citations aren't yielded here — the caller
    already has `chunks` (the same list passed in) to display alongside
    the streamed text once it completes.

    Raises:
        LLMError: if the API call fails, including mid-stream (e.g. a
            connection drop) or if Claude refuses the query.
    """
    context = build_context(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {query}"

    client = _get_client()
    with _translate_anthropic_errors():
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            yield from stream.text_stream
            final_message = stream.get_final_message()

    if final_message.stop_reason == "refusal":
        raise LLMError("Claude declined to answer this query.")
