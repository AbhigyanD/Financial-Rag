"""Structured JSON logs: request ID, per-stage latency, tokens, estimated cost.

Why: one JSON line per event makes any request traceable end to end, and
cost is computed from the token counts the APIs actually report — with
prices read from pricing.toml and always labeled as an estimate.

Usage: the API opens a Trace per request; pipeline code wraps work in
`stage("name")` and reports usage with `add_tokens(...)`. Both are no-ops
when no Trace is active (tests, scripts), so pipeline modules never need
to know whether they're being observed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import tomllib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator

from financial_rag.config import settings

logger = logging.getLogger("financial_rag")


def configure_logging() -> None:
    """Send one JSON object per line to stderr. Idempotent."""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, default=str))


@dataclass
class Trace:
    request_id: str
    route: str
    started: float = field(default_factory=time.perf_counter)
    stages_ms: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(
        default_factory=lambda: {"embedding": 0, "llm_input": 0, "llm_output": 0}
    )
    extra: dict = field(default_factory=dict)


_current: ContextVar[Trace | None] = ContextVar("financial_rag_trace", default=None)


def current_trace() -> Trace | None:
    return _current.get()


@contextmanager
def activate(trace: Trace) -> Iterator[Trace]:
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a block and attribute it to the active request (if any)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        trace = _current.get()
        if trace is not None:
            ms = (time.perf_counter() - t0) * 1000
            trace.stages_ms[name] = round(trace.stages_ms.get(name, 0.0) + ms, 1)


def add_tokens(**counts: int) -> None:
    trace = _current.get()
    if trace is not None:
        for key, value in counts.items():
            trace.tokens[key] = trace.tokens.get(key, 0) + int(value)


def note(**fields) -> None:
    """Attach extra fields (e.g. abstain_reason) to the active request's log line."""
    trace = _current.get()
    if trace is not None:
        trace.extra.update(fields)


@lru_cache(maxsize=1)
def _prices() -> dict:
    try:
        with open(settings.pricing_file, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        log_event("pricing_unavailable", level=logging.WARNING, path=settings.pricing_file, error=str(e))
        return {}


def estimate_cost_usd(tokens: dict[str, int]) -> dict[str, float | None]:
    """Cost estimate per component. None means the model has no price in
    pricing.toml — reported as unknown rather than silently as $0."""
    prices = _prices()
    llm = prices.get("llm", {}).get(settings.llm_model)
    emb = prices.get("embedding", {}).get(settings.embedding_model)

    llm_cost = None
    if llm is not None:
        llm_cost = (
            tokens.get("llm_input", 0) * llm["input_per_mtok"]
            + tokens.get("llm_output", 0) * llm["output_per_mtok"]
        ) / 1_000_000
    emb_cost = None if emb is None else tokens.get("embedding", 0) * emb["input_per_mtok"] / 1_000_000

    total = None if llm_cost is None or emb_cost is None else llm_cost + emb_cost
    return {
        "embedding": None if emb_cost is None else round(emb_cost, 8),
        "llm": None if llm_cost is None else round(llm_cost, 8),
        "total": None if total is None else round(total, 8),
    }


def finish(trace: Trace, status: int, **fields) -> dict:
    """Emit the one summary line for a request and return it."""
    record = {
        "request_id": trace.request_id,
        "route": trace.route,
        "status": status,
        "total_ms": round((time.perf_counter() - trace.started) * 1000, 1),
        "stages_ms": trace.stages_ms,
        "tokens": trace.tokens,
        "est_cost_usd": estimate_cost_usd(trace.tokens),
        "cost_is_estimate": True,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        **trace.extra,
        **fields,
    }
    log_event("request_complete", **record)
    return record
