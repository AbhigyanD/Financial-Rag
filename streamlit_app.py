"""Streamlit frontend for the Financial RAG API.

Why: a thin client — it only calls the HTTP API (never imports the
pipeline), so it can be deployed and scaled separately from the backend.

Run the backend first, then this app:
    uv run uvicorn financial_rag.api:app --reload
    uv run streamlit run streamlit_app.py

Configure the backend URL via the API_URL environment variable if it's
not running on the default localhost:8000.
"""

from __future__ import annotations

import html
import json
import os
from urllib.parse import quote

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Financial RAG", layout="wide")

# --- Styling ------------------------------------------------------------
# Streamlit's dark theme (.streamlit/config.toml) sets the base palette;
# this layer adds the typography and detail work — monospace tags for
# technical values (filenames, page numbers, scores), pill-shaped
# buttons/inputs, and quiet borders instead of colored alert boxes.

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    code, .mono {
        font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    }

    /* Tighten the default Streamlit chrome */
    .block-container {
        padding-top: 2.5rem;
        max-width: 900px;
    }

    /* App title */
    .app-title {
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: #f5f5f5;
        margin-bottom: 0.15rem;
    }
    .app-subtitle {
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }

    /* Sidebar section labels */
    .sidebar-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #777;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
    }

    /* Monospace pill — used for source/page/model tags, like an inline code chip */
    .tag {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        color: #ccc;
        padding: 0.1rem 0.5rem;
        border-radius: 5px;
        margin-right: 0.35rem;
    }

    /* Citation card */
    .citation {
        border: 1px solid #222;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.5rem;
        background: #0f0f0f;
    }
    .citation-text {
        color: #aaa;
        font-size: 0.85rem;
        margin-top: 0.4rem;
        line-height: 1.45;
    }

    /* Pill buttons */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 999px;
        border: 1px solid #333;
        background: #161616;
        color: #eee;
        font-weight: 500;
        padding: 0.35rem 1.1rem;
    }
    .stButton > button:hover {
        border-color: #555;
        background: #1e1e1e;
        color: #fff;
    }

    /* Metric */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }

    hr {
        border-color: #1e1e1e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _api_get(path: str) -> dict | None:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Couldn't reach the API at {API_URL}: {e}")
        return None


def _stream_query(query: str, top_k: int, source: str | None):
    """Yield (event, data) pairs by parsing the /query/stream SSE response.

    A tiny hand-rolled SSE parser rather than a library — the wire format
    here is fixed (we control both ends): each event is exactly one
    "event: <name>" line followed by one "data: <json>" line, blank-line
    terminated.
    """
    payload = {"query": query, "top_k": top_k}
    if source:
        payload["source"] = source

    with requests.post(
        f"{API_URL}/query/stream", json=payload, stream=True, timeout=120
    ) as response:
        response.raise_for_status()
        event_name = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None or raw_line == "":
                continue
            if raw_line.startswith("event: "):
                event_name = raw_line[len("event: ") :]
            elif raw_line.startswith("data: "):
                data = json.loads(raw_line[len("data: ") :])
                yield event_name, data


ABSTAIN_REASONS = {
    "weak_retrieval": "No uploaded passage was close enough to the question to answer from.",
    "model_found_no_support": "The model read the closest passages and found no answer in them.",
    "no_valid_citations": "The model's answer couldn't be traced to any retrieved passage, so it was withheld.",
}


def _error_message(e: requests.RequestException) -> str:
    """Pull the API's {"error": {"message", "request_id"}} out of a failed call."""
    response = getattr(e, "response", None)
    if response is None:
        return f"Couldn't reach the API at {API_URL}."
    try:
        err = response.json()["error"]
        return f"{err['message']} (request {err['request_id']})"
    except (ValueError, KeyError, TypeError):
        return f"HTTP {response.status_code}: {response.text[:200]}"


def _render_citations(citations: list[dict]) -> None:
    # Document text is untrusted: escape it before it goes into raw HTML,
    # or a PDF containing markup would render (or run) in the page.
    st.markdown("**Sources**")
    for c in citations:
        st.markdown(
            f'<div class="citation">'
            f'<span class="tag">[{c["id"]}]</span>'
            f'<span class="tag">{html.escape(c["source"])}</span>'
            f'<span class="tag">page {c["page_number"]}</span>'
            f'<span class="tag">similarity {c["similarity"]:.2f}</span>'
            f'<div class="citation-text">{html.escape(c["text"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_answer(message: dict) -> None:
    if message.get("error"):
        st.error(message["error"])
        return
    if message.get("abstained"):
        st.info(f"**Not found in the provided documents.**  \n{ABSTAIN_REASONS.get(message.get('abstain_reason'), '')}")
        return
    st.markdown(message["content"])
    if message.get("invalid_citation_ids"):
        ids = ", ".join(f"[{i}]" for i in message["invalid_citation_ids"])
        st.caption(f"Removed citation(s) {ids}: they pointed at no retrieved passage.")
    if message.get("citations"):
        _render_citations(message["citations"])


# --- Sidebar: document upload + index status -----------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-label">Documents</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload", type=["pdf", "txt"], label_visibility="collapsed"
    )
    if uploaded_file is not None and st.button("Ingest document", use_container_width=True):
        with st.spinner(f"Processing {uploaded_file.name}"):
            try:
                response = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
                st.success(
                    f"{result['source']} — {result['pages']} pages, "
                    f"{result['chunks_stored']} chunks stored."
                )
                if result["empty_pages"]:
                    st.warning(
                        f"{result['empty_pages']} page(s) had no extractable text "
                        "(likely scanned images) and can't be searched."
                    )
            except requests.RequestException as e:
                st.error(f"Ingestion failed: {_error_message(e)}")

    st.divider()

    count = _api_get("/documents/count")
    if count is not None:
        st.metric("Chunks indexed", count["total_chunks"])

    st.divider()
    st.markdown('<div class="sidebar-label">Retrieval</div>', unsafe_allow_html=True)
    source_filter = st.text_input(
        "Restrict to source", placeholder="e.g. quarterly_report.pdf"
    )
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

    st.divider()
    st.markdown('<div class="sidebar-label">Manage</div>', unsafe_allow_html=True)
    if st.button("Remove a document", use_container_width=True):
        st.session_state["show_delete"] = not st.session_state.get("show_delete", False)
    if st.session_state.get("show_delete"):
        delete_source_name = st.text_input("Filename to remove", key="delete_source_input")
        if st.button("Confirm delete", use_container_width=True) and delete_source_name:
            try:
                response = requests.delete(
                    f"{API_URL}/documents/{quote(delete_source_name, safe='')}", timeout=30
                )
                response.raise_for_status()
                result = response.json()
                st.success(f"Deleted {result['chunks_deleted']} chunks for {result['source']}.")
            except requests.RequestException as e:
                st.error(f"Delete failed: {_error_message(e)}")


# --- Main: chat interface -------------------------------------------------

st.markdown('<div class="app-title">Financial RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask questions grounded in the documents you\'ve uploaded. '
    "Answers cite numbered passages; uncited answers are withheld.</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=None):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            _render_answer(message)

if user_query := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=None):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=None):
        placeholder = st.empty()
        streamed = ""
        message: dict = {"role": "assistant", "content": ""}

        try:
            for event, data in _stream_query(user_query, top_k, source_filter or None):
                if event == "delta":
                    streamed += data
                    placeholder.markdown(streamed + "▌")
                elif event == "final":
                    # The final event is the validated answer: it replaces
                    # the raw streamed text (invalid citations removed, or
                    # an abstention instead).
                    message.update(
                        content=data["answer"],
                        abstained=data["abstained"],
                        abstain_reason=data["abstain_reason"],
                        citations=data["citations"],
                        invalid_citation_ids=data["invalid_citation_ids"],
                    )
                elif event == "error":
                    message["error"] = f"{data['message']} (request {data['request_id']})"
        except requests.RequestException as e:
            message["error"] = _error_message(e)

        placeholder.empty()
        _render_answer(message)

    st.session_state.messages.append(message)
