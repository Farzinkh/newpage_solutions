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
from app.ingestion.extractor import extract_items
from app.ingestion.item_store import InMemoryItemStore
from app.ingestion.redactor import preprocess
from app.interfaces import Embedder, VectorStore
from app.logging_config import log_event
from app.models import MeetingBrief, Turn

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        item_store: InMemoryItemStore,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._item_store = item_store

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

        # Tag decisions / action items for aggregation queries (idempotent per
        # meeting). Runs on the redacted turns so nothing sensitive is stored.
        items = (
            extract_items(meeting_id, clean_turns)
            if self._settings.extraction_enabled
            else []
        )
        self._item_store.replace(meeting_id, items)

        # Meeting brief (highlights): whole-meeting context injected at query
        # time so answers aren't limited to the isolated top-k chunks. Preserves
        # first-seen speaker order rather than sorting, so it reads like the room.
        seen: dict[str, None] = {}
        for t in clean_turns:
            seen.setdefault(t.speaker, None)
        brief = MeetingBrief(
            meeting_id=meeting_id,
            participants=list(seen),
            decisions=[it.text for it in items if it.kind == "decision"],
            action_items=[it.text for it in items if it.kind == "action_item"],
        )
        self._item_store.set_brief(meeting_id, brief)

        # Log only redaction *counts*, never the raw PII values.
        log_event(
            logger,
            "ingested_meeting",
            meeting_id=meeting_id,
            turns=len(clean_turns),
            chunks=len(chunks),
            items=len(items),
            redactions=redaction_totals,
        )
        return {
            "meeting_id": meeting_id,
            "turns": len(clean_turns),
            "chunks": len(chunks),
            "items": len(items),
            "redactions": redaction_totals,
        }
