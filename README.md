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

## Next Stages (Planned)

3. **Embeddings** — Convert chunk text to vectors (OpenAI, Anthropic, local model)
4. **Vector Storage** — Store chunks + embeddings in a database (Chroma, Pinecone, etc.)
5. **Retrieval** — Find relevant chunks for a query via similarity search
6. **LLM Response** — Pass retrieved chunks to Claude to generate answers
7. **API Layer** — FastAPI or Streamlit interface for document upload and querying

## File Structure

```
src/financial_rag/
├── __init__.py
└── loaders/
    ├── __init__.py
    ├── document_loader.py    # Stage 1: Extract text
    └── chunker.py            # Stage 2: Split into chunks
```

## Data Flow

```
Upload (PDF/TXT)
    ↓
[Stage 1: document_loader]
    load_document() / load_pdf() / load_text()
    ↓
list[PageText] {page_number, text}
    ↓
[Stage 2: chunker]
    chunk_pages()
    ↓
list[Chunk] {page_number, text, chunk_index}
    ↓
[Stages 3+: Embeddings, Storage, Retrieval, LLM]
    ↓
Answer with citations
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