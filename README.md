# Financial RAG

A retrieval-augmented generation (RAG) system for financial documents. This project builds a pipeline to ingest financial PDFs and plain-text files, extract and chunk their content, and make them retrievable for question-answering with Claude.

## Architecture

The system is built in modular stages, each with a clear responsibility:

### Stage 1: Document Loading (`src/financial_rag/loaders/document_loader.py`)

**Purpose:** Extract raw text from uploaded files (PDF or plain text), preserving page numbers for citations.

**Key Functions:**

- **`load_pdf(file_path_or_bytes) → list[PageText]`**
  - Opens a PDF (from a file path, raw bytes, or file-like object)
  - Extracts text page-by-page using PyMuPDF (`fitz`)
  - Returns one `PageText` entry per page, even if the page is empty (scanned image, no OCR)
  - Cleans up PDF extraction noise (3+ consecutive blank lines → single paragraph break)

- **`load_text(file_path_or_bytes) → list[PageText]`**
  - Reads plain `.txt` files with UTF-8 and latin-1 fallback decoding
  - Returns a single "page" (page_number: 1) for uniform handling downstream

- **`load_document(file_path_or_bytes, filename=None) → list[PageText]`**
  - Dispatcher: routes to `load_pdf` or `load_text` based on file extension
  - Accepts explicit `filename=` for cases where the input is raw bytes with no metadata

- **`get_preview(pages, length=500) → str`**
  - Returns a preview snippet (up to `length` chars) for UI display

- **`get_total_length(pages) → int`**
  - Returns total character count across all pages

**Data Structure:**

```python
class PageText(TypedDict):
    page_number: int | None
    text: str
```

**Example Usage:**

```python
from src.financial_rag.loaders.document_loader import load_document, get_preview, get_total_length

# Load a file (path, bytes, or UploadFile)
pages = load_document("financial_report.pdf")

# Inspect it
print(f"Total chars: {get_total_length(pages)}")
print(f"Preview: {get_preview(pages, length=200)}")
```

### Stage 2: Chunking (`src/financial_rag/loaders/chunker.py`)

**Purpose:** Split extracted text into smaller, overlapping chunks suitable for embedding and retrieval, while preserving page numbers for citations.

**Key Functions:**

- **`chunk_pages(pages, max_chunk_size=1000, overlap=False) → list[Chunk]`**
  - Takes `list[PageText]` from stage 1
  - Splits each page into paragraphs (on blank lines)
  - Greedily merges paragraphs until reaching `max_chunk_size` (in characters)
  - Assigns a global `chunk_index` for sequencing
  - Each chunk preserves its source `page_number` for later citations

- **`count_chunks(pages, max_chunk_size=1000) → int`**
  - Quick estimate of how many chunks will be produced (for UI progress reporting)

**Data Structure:**

```python
class Chunk(TypedDict):
    page_number: int      # Source page (1-indexed)
    text: str            # Chunk content
    chunk_index: int     # Global sequence number (0-indexed)
```

**Strategy:**

1. Split each page on paragraph boundaries (`\n\n`)
2. Greedily merge paragraphs within the same page until adding another would exceed `max_chunk_size`
3. Each chunk is assigned a global index and tagged with its source page number

This approach:
- Respects document structure (no mid-sentence breaks)
- Keeps chunks under a predictable size for embedding
- Preserves page numbers for citation and traceability

**Example Usage:**

```python
from src.financial_rag.loaders.document_loader import load_document
from src.financial_rag.loaders.chunker import chunk_pages, count_chunks

# Load and chunk
pages = load_document("quarterly_report.pdf")
chunks = chunk_pages(pages, max_chunk_size=1000)

print(f"Created {len(chunks)} chunks")

for chunk in chunks:
    print(f"Chunk {chunk['chunk_index']} (page {chunk['page_number']}): {len(chunk['text'])} chars")
```

## Installation

```bash
# Install the project with dependencies
uv sync

# Or via pip
pip install -e .
```

**Dependencies:**

- `pymupdf>=1.24` — For PDF text extraction
- Python 3.11+

## Testing

Both modules include comprehensive smoke tests:

```bash
# Test document_loader
uv run python3 -c "
from src.financial_rag.loaders.document_loader import load_pdf, load_text
import pymupdf as fitz

# Create a test PDF
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), 'Test content')
pdf_bytes = doc.tobytes()
doc.close()

# Load it
pages = load_pdf(pdf_bytes)
print(f'✓ Loaded {len(pages)} pages')
"

# Test chunker
uv run python3 -c "
from src.financial_rag.loaders.document_loader import load_text
from src.financial_rag.loaders.chunker import chunk_pages

pages = load_text(b'Paragraph one.\n\nParagraph two.')
chunks = chunk_pages(pages, max_chunk_size=100)
print(f'✓ Created {len(chunks)} chunks')
"
```

## Error Handling

Both stages use custom exceptions for clear error reporting:

- **`DocumentLoadError`** — Raised when a file cannot be opened or parsed (corrupt PDF, unsupported format, encoding issues)
- **`ChunkingError`** — Raised during chunk processing (reserved for future use; currently all failures are silent)

Example:

```python
from src.financial_rag.loaders.document_loader import load_pdf, DocumentLoadError

try:
    pages = load_pdf(b"not a real pdf")
except DocumentLoadError as e:
    print(f"Failed to load PDF: {e}")
```

### Stage 3: Embeddings (`src/financial_rag/embeddings.py`)

**Purpose:** Convert chunk text into vector embeddings using OpenAI's `text-embedding-3-small` (1536 dimensions).

**Key Functions:**

- **`embed_chunks(chunks, batch_size=100) → list[EmbeddedChunk]`** — embeds a list of chunks, batched
- **`embed_query(query) → list[float]`** — embeds a query string with the same model, for similarity comparison
- **`get_embedding_dimensions() → int`** — returns the model's vector size (for storage schema validation)

Requires `OPENAI_API_KEY` in the environment.

### Stage 4: Vector Storage (`src/financial_rag/storage.py`)

**Purpose:** Persist embedded chunks in a local Chroma vector database and support similarity search.

**Key Functions:**

- **`store_chunks(embedded_chunks, source) → int`** — upserts chunks (re-ingesting a document overwrites, doesn't duplicate)
- **`query_chunks(query_embedding, top_k=5, source=None) → list[StoredChunk]`** — raw similarity search (Chroma cosine distance)
- **`delete_source(source) → int`** — removes all chunks for a document
- **`count_stored_chunks() → int`** — total indexed chunks

Data persists to `./chroma_data/` (gitignored) using cosine distance.

### Stage 5: Retrieval (`src/financial_rag/retrieval.py`)

**Purpose:** Glue stage 3 + stage 4 together — embed a query, search the store, and return results with a normalized `similarity` score in `[0, 1]` (higher = more relevant), rather than exposing Chroma's raw distance.

**Key Function:**

- **`retrieve(query, top_k=5, source=None) → list[RetrievedChunk]`**

### Stage 6: LLM Response (`src/financial_rag/llm.py`)

**Purpose:** Build a grounded prompt from retrieved chunks and call Claude (`claude-opus-5` via the Messages API) to generate a cited answer.

**Key Functions:**

- **`generate_answer(query, chunks) → Answer`** — returns `{text, citations}`; the system prompt instructs Claude to answer only from the provided context, cite `[source, page N]` inline, and say so explicitly when the context is insufficient.
- **`build_context(chunks) → str`** — formats chunks into a labeled context block

Requires `ANTHROPIC_API_KEY` in the environment.

### Stage 7: API Layer (`src/financial_rag/api.py`)

**Purpose:** Expose the full pipeline over HTTP with FastAPI.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Upload a PDF/TXT file — runs stages 1–4 |
| `DELETE` | `/documents/{source}` | Remove an ingested document |
| `GET` | `/documents/count` | Total chunks indexed |
| `POST` | `/query` | Ask a question — runs stages 5–6, returns answer + citations |
| `GET` | `/health` | Liveness check |

Run locally:

```bash
uv run uvicorn financial_rag.api:app --reload
```

Then visit `http://localhost:8000/docs` for interactive API docs (Swagger UI).

## File Structure

```
src/financial_rag/
├── __init__.py
├── api.py                 # Stage 7: FastAPI endpoints
├── llm.py                 # Stage 6: Claude answer generation
├── retrieval.py            # Stage 5: Query embedding + search
├── storage.py              # Stage 4: Chroma vector store
├── embeddings.py           # Stage 3: OpenAI embeddings
└── loaders/
    ├── __init__.py
    ├── document_loader.py    # Stage 1: Extract text
    └── chunker.py            # Stage 2: Split into chunks
```

## Data Flow

```
Upload (PDF/TXT)
    ↓
[Stage 1: document_loader] load_document()
    ↓
list[PageText] {page_number, text}
    ↓
[Stage 2: chunker] chunk_pages()
    ↓
list[Chunk] {page_number, text, chunk_index}
    ↓
[Stage 3: embeddings] embed_chunks()
    ↓
list[EmbeddedChunk] {..., embedding}
    ↓
[Stage 4: storage] store_chunks()  →  Chroma (./chroma_data/)

User query
    ↓
[Stage 5: retrieval] retrieve()  →  embed_query() + query_chunks()
    ↓
list[RetrievedChunk] {..., similarity}
    ↓
[Stage 6: llm] generate_answer()  →  Claude (claude-opus-5)
    ↓
Answer with citations
    ↓
[Stage 7: api] exposed over HTTP (FastAPI)
```

## Notes

- **PDF Page Alignment:** Empty pages (scans without OCR) are preserved in the output to keep page numbers aligned with the actual PDF page count.
- **Whitespace Handling:** PDF extraction noise (3+ blank lines) is cleaned during load; everything else is preserved to maintain paragraph structure for chunking.
- **Character-Based Chunking:** Currently uses character count for chunk size. Token-based chunking (for exact model context limits) can be added as an option in stage 3.
- **Page Number Tracking:** Each chunk knows its source page, enabling citation footnotes in the final answer (e.g., "According to page 3...").

## Contributing

To add a new loader format (e.g., `.docx`), extend `document_loader.py`:

1. Add a `load_docx()` function that returns `list[PageText]`
2. Update `load_document()` to dispatch `.docx` → `load_docx()`
3. Add tests in the smoke-test block

To adjust chunking strategy, modify `chunker.py`:

1. Update `_merge_chunks()` to use a different merging algorithm (e.g., sentence-aware, token-aware)
2. Add a new parameter to `chunk_pages()` (e.g., `strategy="character"` or `strategy="token"`)

## Progress Log

### 2026-09-14

- Implemented **Stage 3 (Embeddings)** — `embeddings.py`, using OpenAI's `text-embedding-3-small`.
- Implemented **Stage 4 (Vector Storage)** — `storage.py`, using a local Chroma collection (cosine distance) with upsert-based ingestion and per-document deletion.
- Implemented **Stage 5 (Retrieval)** — `retrieval.py`, gluing query embedding + vector search together and normalizing Chroma's raw distance into a `[0, 1]` similarity score.
- Implemented **Stage 6 (LLM Response)** — `llm.py`, prompting Claude (`claude-opus-5`) to answer strictly from retrieved context with inline `[source, page N]` citations.
- Implemented **Stage 7 (API Layer)** — `api.py`, a FastAPI app exposing document upload/delete/count and query endpoints over the full pipeline.
- Added a `tests/` suite (pytest) covering the loader, chunker, retrieval similarity math, and LLM prompt formatting — all offline, no API keys required.
- Fixed a stale CI workflow (wrong Python versions, missing dependency install, no tests to run) to actually use `uv sync` + `uv run pytest` against Python 3.11.
- General repo cleanup: added `.gitignore`, removed committed `__pycache__` bytecode, removed a stray unused virtualenv, fixed the VS Code interpreter path.

## AI Usage

Claude (via Claude Code) was used throughout this project to:

- **Structure the codebase** — proposing the module/file layout for each pipeline stage (`embeddings.py`, `storage.py`, `retrieval.py`, `llm.py`, `api.py`) consistent with the existing loader/chunker pattern.
- **Write comments and docstrings** — explaining design decisions (e.g., why upsert instead of add, why cosine distance over Chroma's default L2, why retrieval normalizes scores before returning them).
- **Debug issues** — catching a `return` vs `raise` bug in an exception path, a batch-accumulation bug in `embed_chunks` that silently dropped all but the last batch, a missing `return` statement, and a stale CI workflow that referenced non-existent files and Python versions.
- **Come up with test ideas** — suggesting which pure-logic paths were safe/valuable to test without hitting paid APIs (e.g., distance-to-similarity conversion, prompt context formatting) and edge cases to cover (empty input, out-of-range values, multi-excerpt formatting).

All code was reviewed before being committed; the developer made the final calls on architecture decisions (e.g., embedding provider, vector store choice, similarity score convention).