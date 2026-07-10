"""Ingestion orchestration.

Both input sources (uploaded ``.txt`` and voice) converge here as ``list[Turn]``.
The pipeline then runs one shared path: preprocess (clean + redact) each turn,
chunk, embed, and upsert into the vector store. Idempotency comes from the
stable chunk ids produced by the chunker.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.ingestion.chunker import chunk_turns
from app.ingestion.redactor import preprocess
from app.interfaces import Embedder, VectorStore
from app.logging_config import log_event
from app.models import Turn

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store

    def ingest_turns(self, meeting_id: str, turns: list[Turn]) -> dict[str, object]:
        redaction_totals: dict[str, int] = {}
        clean_turns: list[Turn] = []
        for turn in turns:
            text, counts = preprocess(turn.text, redact=self._settings.redaction_enabled)
            if not text:
                continue
            for k, v in counts.items():
                redaction_totals[k] = redaction_totals.get(k, 0) + v
            clean_turns.append(turn.model_copy(update={"text": text}))

        chunks = chunk_turns(meeting_id, clean_turns, self._settings.max_chunk_chars)
        if chunks:
            vectors = self._embedder.embed([c.text for c in chunks])
            self._store.upsert(chunks, vectors)

        # Log only redaction *counts*, never the raw PII values.
        log_event(
            logger,
            "ingested_meeting",
            meeting_id=meeting_id,
            turns=len(clean_turns),
            chunks=len(chunks),
            redactions=redaction_totals,
        )
        return {
            "meeting_id": meeting_id,
            "turns": len(clean_turns),
            "chunks": len(chunks),
            "redactions": redaction_totals,
        }
