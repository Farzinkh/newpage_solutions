"""HTTP API.

Thin transport layer over `Services`. Endpoints:
  POST /ingest   — text (from file or voice) -> normalise -> pipeline
  POST /query    — question -> answer + citations + retrieval debug
  GET  /meetings — ingested meeting ids
  GET  /health   — liveness + corpus size

Services are built once from config and injected via a dependency, so tests can
override them with a fake-backed instance.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.factory import Services, build_services
from app.logging_config import configure_logging
from app.models import Answer, ExtractedItem, HistoryTurn, MeetingBrief


class IngestRequest(BaseModel):
    meeting_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: Literal["file", "voice"] = "file"
    # Meeting start wall-clock time. When set, each turn's relative timestamp is
    # anchored to an absolute datetime so citations disambiguate across meetings.
    started_at: datetime | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    meeting_id: str | None = None
    # Prior turns for follow-up resolution; optional so single-shot still works.
    history: list[HistoryTurn] = Field(default_factory=list)


@lru_cache
def get_services() -> Services:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    return build_services(settings)


app = FastAPI(title="Meeting Intelligence")


@app.get("/health")
def health(services: Services = Depends(get_services)) -> dict[str, object]:
    return {"status": "ok", "chunks": services.vector_store.count()}


@app.get("/meetings")
def meetings(services: Services = Depends(get_services)) -> dict[str, list[str]]:
    return {"meetings": services.vector_store.list_meetings()}


@app.post("/ingest")
def ingest(req: IngestRequest, services: Services = Depends(get_services)) -> dict[str, object]:
    transcriber = (
        services.voice_transcriber if req.source == "voice" else services.file_transcriber
    )
    turns = transcriber.to_turns(req.text)
    if not turns:
        # Fail loudly rather than silently ingesting nothing — the usual cause is
        # a transcript whose format we don't recognise (no speaker labels).
        raise HTTPException(
            status_code=422,
            detail="No speaker turns parsed. Expected '[HH:MM:SS] Speaker: text' "
            "or 'Speaker: text' lines.",
        )
    return services.ingestion.ingest_turns(req.meeting_id, turns, req.started_at)


@app.post("/query")
def query(req: QueryRequest, services: Services = Depends(get_services)) -> Answer:
    return services.query(req.question, req.meeting_id, req.history)


@app.get("/items")
def items(
    meeting_id: str | None = None,
    kind: Literal["action_item", "decision"] | None = None,
    services: Services = Depends(get_services),
) -> dict[str, list[ExtractedItem]]:
    return {"items": services.item_store.list(meeting_id, kind)}


@app.get("/brief")
def brief(
    meeting_id: str, services: Services = Depends(get_services)
) -> MeetingBrief:
    b = services.item_store.get_brief(meeting_id)
    if b is None:
        raise HTTPException(status_code=404, detail=f"No brief for meeting '{meeting_id}'")
    return b
