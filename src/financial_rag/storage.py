"""Store embedded chunks in a vector database and query them by similarity.

This is stage 4 of the ingest pipeline: given a list[EmbeddedChunk] from
stage 3 (embeddings), persist them in a vector store so stage 5 (retrieval)
can find the most relevant chunks for a user's query.

We use Chroma (local, embedded, no server to run) as the default backend.
It stores vectors + metadata + text together and handles the similarity
search (cosine/L2) internally, so this module is mostly a thin, storage-
agnostic wrapper: swapping Chroma for Pinecone/Weaviate later should only
require changes inside this file, not in embeddings.py or the caller.

Design notes:
- One Chroma "collection" per logical document set. Keep it simple for now:
  a single collection name, e.g. "financial_documents".
- Chroma requires a unique string `id` per vector. Use something derived
  from source filename + chunk_index so re-ingesting the same file is
  idempotent (upsert) rather than creating duplicates.
- Metadata (page_number, chunk_index, filename) must be stored alongside
  the vector so retrieval results can be cited back to a page.
- Keep query results as plain dicts/lists, not Chroma-specific types, so
  stage 5/6 don't need to know which vector DB is underneath.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

import chromadb
from chromadb.errors import ChromaError

from financial_rag.embeddings import EmbeddedChunk, get_embedding_dimensions


class StoredChunk(TypedDict):
    """A chunk retrieved from the vector store, with its similarity score."""

    page_number: int
    text: str
    chunk_index: int
    score: float  # similarity/distance score (lower or higher = closer, backend-dependent)
    source: str  # filename or document identifier this chunk came from


class StorageError(Exception):
    """Raised when a vector store operation fails."""


# TODO: pick a persistence location, e.g.:
#   PERSIST_DIRECTORY = "./chroma_data"
# Chroma will create/reuse this directory on disk so the index survives
# across process restarts (no need to re-embed documents every run).
PERSIST_DIRECTORY = "./chroma_data"
COLLECTION_NAME = "financial_documents"


@lru_cache(maxsize=1)
def _get_client() -> chromadb.ClientAPI:
    """Create/return the Chroma persistent client, cached for the process."""
    try:
        return chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    except ChromaError as e:
        raise StorageError(f"Failed to open Chroma store at '{PERSIST_DIRECTORY}': {e}") from e


def _get_collection() -> chromadb.Collection:
    """Create/return the Chroma collection used to store chunks."""
    client = _get_client()
    try:
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except ChromaError as e:
        raise StorageError(f"Failed to get/create collection '{COLLECTION_NAME}': {e}") from e


def _make_chunk_id(source: str, chunk_index: int) -> str:
    """Build a stable, unique id for one chunk.

    Deterministic: the same (source, chunk_index) pair always yields the
    same id, so re-ingesting a document upserts existing vectors instead
    of duplicating them. Path separators are replaced since they have no
    special meaning to Chroma but keep ids readable/debuggable as a single
    token (e.g. in logs).
    """
    safe_source = source.replace("/", "_").replace("\\", "_")
    return f"{safe_source}::{chunk_index}"


def store_chunks(embedded_chunks: list[EmbeddedChunk], source: str) -> int:
    """Store embedded chunks in the vector database.

    Args:
        embedded_chunks: List of EmbeddedChunk dicts from embeddings.py.
        source: Identifier for the document these chunks came from (e.g.
            the original filename). Stored as metadata for citations and
            used to build stable chunk ids.

    Returns:
        The number of chunks stored.
    """
    if not embedded_chunks:
        return 0

    ids = [_make_chunk_id(source, c["chunk_index"]) for c in embedded_chunks]
    embeddings = [c["embedding"] for c in embedded_chunks]
    documents = [c["text"] for c in embedded_chunks]
    metadatas = [
        {
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"],
            "source": source,
        }
        for c in embedded_chunks
    ]

    collection = _get_collection()
    try:
        # upsert (not add) so re-ingesting the same document overwrites
        # existing vectors instead of erroring or duplicating rows.
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
    except ChromaError as e:
        raise StorageError(
            f"Failed to store {len(embedded_chunks)} chunks for '{source}': {e}"
        ) from e

    return len(embedded_chunks)


def query_chunks(
    query_embedding: list[float], top_k: int = 5, source: str | None = None
) -> list[StoredChunk]:
    """Find the top_k most similar chunks to a query embedding.

    Args:
        query_embedding: Vector from embeddings.embed_query(), same model
            as the stored chunks.
        top_k: Number of results to return.
        source: If given, restrict the search to chunks from this document
            only (via Chroma's `where` metadata filter).

    Returns:
        A list of StoredChunk dicts ordered by relevance (best match first).
        `score` is Chroma's raw distance — LOWER means more similar, not
        higher. Invert/normalize downstream if a "higher is better" score
        is more convenient for the retrieval/ranking stage.
    """
    collection = _get_collection()
    where = {"source": source} if source is not None else None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
    except ChromaError as e:
        raise StorageError(f"Query failed: {e}") from e

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    return [
        StoredChunk(
            page_number=metadata["page_number"],
            text=document,
            chunk_index=metadata["chunk_index"],
            score=distance,
            source=metadata["source"],
        )
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def delete_source(source: str) -> int:
    """Delete all chunks belonging to a given source document.

    Useful for re-ingesting an updated version of a document, or removing
    one that's no longer needed.

    Returns:
        The number of chunks deleted.
    """
    collection = _get_collection()
    try:
        # Look up matching ids first so we can report a count — delete()
        # itself doesn't return how many rows it removed.
        matches = collection.get(where={"source": source})
        matching_ids = matches["ids"]
        if not matching_ids:
            return 0
        collection.delete(ids=matching_ids)
    except ChromaError as e:
        raise StorageError(f"Failed to delete chunks for '{source}': {e}") from e

    return len(matching_ids)


def count_stored_chunks() -> int:
    """Return the total number of chunks currently in the vector store.

    Useful for a UI/CLI to report ingestion progress or index size.
    """
    collection = _get_collection()
    try:
        return collection.count()
    except ChromaError as e:
        raise StorageError(f"Failed to count stored chunks: {e}") from e
