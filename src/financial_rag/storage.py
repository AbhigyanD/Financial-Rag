"""Store embedded chunks in a vector database and query them by similarity.

Why: the only module that knows Chroma exists, so swapping vector stores
means changing this file and nothing else.

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

from financial_rag.config import settings
import re

from financial_rag.embeddings import EmbeddedChunk


class StoredChunk(TypedDict):
    """A chunk retrieved from the vector store, with its similarity score."""

    page_number: int
    text: str
    chunk_index: int
    score: float  # similarity/distance score (lower or higher = closer, backend-dependent)
    source: str  # filename or document identifier this chunk came from


class StorageError(Exception):
    """Raised when a vector store operation fails."""


PERSIST_DIRECTORY = settings.persist_directory


def _collection_name() -> str:
    """COLLECTION_NAME plus the embedding model, e.g.
    financial_documents-sentence-transformers-all-MiniLM-L6-v2.

    Vectors from different models (or dimensions) can't be compared, so
    switching EMBEDDING_MODEL starts a fresh collection instead of
    failing on a dimension mismatch or silently mixing incompatible vectors.
    """
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", settings.embedding_model).strip("-._")
    return f"{settings.collection_name}-{slug}"[:512]


COLLECTION_NAME = _collection_name()


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
        # Cosine distance (not Chroma's default squared-L2) keeps scores in
        # a predictable [0, 2] range, which retrieval.py relies on to
        # convert distance into a [0, 1] similarity score. Only applies to
        # a newly created collection — has no effect if one already exists
        # on disk with a different metric.
        return client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
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
    # Chroma rejects a single write above its max batch size (5461 in
    # chromadb 1.5.9), so a large document is written in slices.
    batch = _get_client().get_max_batch_size()
    try:
        for i in range(0, len(ids), batch):
            # upsert (not add) so re-ingesting the same document overwrites
            # existing vectors instead of erroring or duplicating rows.
            collection.upsert(
                ids=ids[i : i + batch],
                embeddings=embeddings[i : i + batch],
                documents=documents[i : i + batch],
                metadatas=metadatas[i : i + batch],
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


def get_all_chunks(source: str | None = None) -> list[dict]:
    """Return every stored chunk's text + metadata (no embeddings).

    Chroma has no native keyword/BM25 index — retrieval.py's hybrid
    search builds one from this at query time. Fine at prototype scale
    (hundreds to a few thousand chunks); a production deployment would
    maintain a persisted/incremental BM25 index instead of rebuilding
    one from a full collection scan on every hybrid query.
    """
    collection = _get_collection()
    where = {"source": source} if source is not None else None
    try:
        result = collection.get(where=where)
    except ChromaError as e:
        raise StorageError(f"Failed to fetch chunks for keyword index: {e}") from e

    ids = result["ids"] or []
    documents = result["documents"] or []
    metadatas = result["metadatas"] or []

    return [
        {
            "id": id_,
            "text": document,
            "page_number": metadata["page_number"],
            "chunk_index": metadata["chunk_index"],
            "source": metadata["source"],
        }
        for id_, document, metadata in zip(ids, documents, metadatas)
    ]


def count_stored_chunks() -> int:
    """Return the total number of chunks currently in the vector store.

    Useful for a UI/CLI to report ingestion progress or index size.
    """
    collection = _get_collection()
    try:
        return collection.count()
    except ChromaError as e:
        raise StorageError(f"Failed to count stored chunks: {e}") from e


def list_sources() -> list[dict]:
    """One row per stored document: name, chunk count, highest page seen."""
    collection = _get_collection()
    try:
        metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    except ChromaError as e:
        raise StorageError(f"Failed to list documents: {e}") from e

    by_source: dict[str, dict] = {}
    for m in metadatas:
        row = by_source.setdefault(m["source"], {"source": m["source"], "chunks": 0, "pages": 0})
        row["chunks"] += 1
        row["pages"] = max(row["pages"], m["page_number"])
    return sorted(by_source.values(), key=lambda r: r["source"].lower())
