from datetime import datetime

from app.config import Settings
from app.factory import build_services
from app.ingestion.timestamps import parse_filename, parse_offset, to_absolute


def test_parse_offset_handles_hms_and_ms():
    assert parse_offset("00:02:36").total_seconds() == 156
    assert parse_offset("02:36").total_seconds() == 156      # MM:SS
    assert parse_offset("01:00:00").total_seconds() == 3600
    assert parse_offset("not-a-time") is None


def test_to_absolute_adds_offset_to_start():
    start = datetime(2026, 6, 3, 15, 48, 0)
    assert to_absolute(start, "00:02:36") == "2026-06-03 15:50:36"


def test_parse_filename_extracts_id_and_start():
    mid, start = parse_filename("product_planning_2026-06-03_1548.txt")
    assert mid == "product_planning"
    assert start == datetime(2026, 6, 3, 15, 48, 0)


def test_parse_filename_without_date_keeps_stem():
    mid, start = parse_filename("hiring_sync.txt")
    assert mid == "hiring_sync"
    assert start is None


def test_ingestion_stamps_absolute_time_on_chunks():
    s = build_services(Settings(embedder_backend="fake", vector_store_backend="memory",
                                reranker_backend="noop", llm_backend="echo"))
    turns = s.file_transcriber.to_turns("[00:02:36] Priya: We ship Q3.")
    s.ingestion.ingest_turns("m1", turns, datetime(2026, 6, 3, 15, 48, 0))
    hit = s.retriever.retrieve("ship Q3", "m1", {})[0].chunk
    assert hit.occurred_at == "2026-06-03 15:50:36"
    assert hit.display_time() == "2026-06-03 15:50:36"


def test_absolute_time_and_meeting_reach_citation():
    s = build_services(Settings(embedder_backend="fake", vector_store_backend="memory",
                                reranker_backend="noop", llm_backend="echo"))
    turns = s.file_transcriber.to_turns("[00:00:05] Priya: Root cause was the pool limit.")
    s.ingestion.ingest_turns("incident_retro", turns, datetime(2026, 6, 5, 10, 15, 0))
    ans = s.query("what was the root cause", "incident_retro")
    assert ans.citations[0].meeting_id == "incident_retro"
    assert ans.citations[0].timestamp == "2026-06-05 10:15:05"
