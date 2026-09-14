"""Pydantic request/response models for the API layer (stage 7).

Kept separate from api.py so route handlers stay focused on
orchestration logic, and the shape of the HTTP contract is easy to find
in one place.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class IngestResponse(BaseModel):
    source: str
    pages: int
    chunks_stored: int


class DeleteResponse(BaseModel):
    source: str
    chunks_deleted: int


class CountResponse(BaseModel):
    total_chunks: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    source: Optional[str] = None


class CitationResponse(BaseModel):
    source: str
    page_number: int
    similarity: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
