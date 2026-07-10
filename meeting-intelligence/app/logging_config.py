"""Structured logging.

Query-time observability is a first-class concern: every query logs the
retrieved chunk ids, similarity + rerank scores, and per-stage latency as a
single JSON line. That is what makes a retrieval black box debuggable in
production — you can answer "why did it return that?" after the fact.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore[attr-defined]
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if as_json else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def log_event(logger: logging.Logger, msg: str, **fields: object) -> None:
    logger.info(msg, extra={"extra_fields": fields})


@contextmanager
def timed(store: dict[str, float], key: str):
    """Record wall-clock ms for a stage into `store[key]`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round((time.perf_counter() - start) * 1000, 1)
