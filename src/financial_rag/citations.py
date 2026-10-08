"""Validate the model's [n] citations against the chunks it was actually sent.

Why: a citation is only trustworthy if it points at text the model really
saw. This parses every [n] marker out of the answer, keeps the ones that
map to an excerpt in the prompt, and strips the rest from the text so the
user never sees a citation that can't be traced to a retrieved chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypedDict

from financial_rag.retrieval import RetrievedChunk

# One bracket holding one or more ids: [3] or [2, 5]. Deliberately narrow,
# so "[Note]" or "[2024]"-style text isn't mistaken for a citation of
# excerpt 2024 — ids above the excerpt count are just reported invalid.
_CITATION = re.compile(r"\[(\d{1,4}(?:\s*,\s*\d{1,4})*)\]")
_CITATION_WITH_SPACE = re.compile(r"([ \t]*)" + _CITATION.pattern)


class Citation(TypedDict):
    id: int  # the [n] the model wrote
    source: str
    page_number: int
    chunk_index: int
    similarity: float
    text: str


@dataclass(frozen=True)
class CitationCheck:
    text: str  # answer with invalid markers removed
    citations: list[Citation]  # valid, deduped, in order of first appearance
    invalid_ids: list[int]  # ids the model wrote that match no sent excerpt


def extract_ids(answer: str) -> list[int]:
    ids = []
    for match in _CITATION.finditer(answer):
        ids.extend(int(part) for part in match.group(1).split(","))
    return ids


def validate_citations(answer: str, sent_chunks: list[RetrievedChunk]) -> CitationCheck:
    """Split the answer's citations into valid and invalid.

    Excerpt [n] is sent_chunks[n-1] — the numbering prompt_builder used.
    """
    valid_range = range(1, len(sent_chunks) + 1)
    invalid: list[int] = []
    cited: dict[int, Citation] = {}

    for cid in extract_ids(answer):
        if cid not in valid_range:
            if cid not in invalid:
                invalid.append(cid)
            continue
        if cid not in cited:
            c = sent_chunks[cid - 1]
            cited[cid] = Citation(
                id=cid,
                source=c["source"],
                page_number=c["page_number"],
                chunk_index=c["chunk_index"],
                similarity=c.get("vector_similarity", c["similarity"]),
                text=c["text"],
            )

    def _rewrite(match: re.Match) -> str:
        space, ids = match.group(1), match.group(2)
        kept = [p.strip() for p in ids.split(",") if int(p) in valid_range]
        # Dropping a whole marker also drops the space before it, so
        # "in 2026 [9]." becomes "in 2026." rather than "in 2026 .".
        return f"{space}[{', '.join(kept)}]" if kept else ""

    cleaned = _CITATION_WITH_SPACE.sub(_rewrite, answer) if invalid else answer
    return CitationCheck(text=cleaned, citations=list(cited.values()), invalid_ids=invalid)
