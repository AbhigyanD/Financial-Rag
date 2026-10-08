"""Pydantic request/response models for the HTTP API.

Why: the API contract lives in one file, and input limits (query length,
top_k range) are enforced here before any pipeline code runs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class IngestResponse(BaseModel):
    source: str
    pages: int
    empty_pages: int  # pages with no extractable text (e.g. scanned, no OCR)
    chunks_stored: int
    request_id: str


class DeleteResponse(BaseModel):
    source: str
    chunks_deleted: int


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    pages: int  # highest page number stored (scanned pages with no text aren't counted)


class InfoResponse(BaseModel):
    llm_provider: str
    llm_model: str
    llm_key_configured: bool
    embedding_provider: str
    embedding_model: str
    hybrid_retrieval: bool
    rerank: bool


class CountResponse(BaseModel):
    total_chunks: int


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    source: Optional[str] = None


class CitationResponse(BaseModel):
    id: int  # the [n] marker used in `answer`
    source: str
    page_number: int
    chunk_index: int
    similarity: float  # cosine similarity rescaled to [0, 1]
    text: str


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: Optional[
        Literal["weak_retrieval", "model_found_no_support", "no_valid_citations"]
    ] = None
    citations: list[CitationResponse]
    invalid_citation_ids: list[int]  # [n] markers the model wrote that matched no excerpt
    request_id: str


class ErrorBody(BaseModel):
    status: int
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
