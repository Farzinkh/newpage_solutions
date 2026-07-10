import pytest
from fastapi.testclient import TestClient

from app.api import app, get_services
from app.config import Settings
from app.factory import build_services

TRANSCRIPT = (
    "[00:00:04] Priya: We commit offline mode for Q3, single device only.\n"
    "[00:00:19] Daniel: Root cause of the outage was a low connection pool limit.\n"
    "[00:00:33] Maya: Reach me at maya@example.com if needed.\n"
    "[00:00:48] Daniel: I'll write the design doc by Friday.\n"
)


@pytest.fixture
def services():
    # Fake profile: no API keys, fully deterministic, no network.
    return build_services(
        Settings(
            embedder_backend="fake",
            vector_store_backend="memory",
            reranker_backend="noop",
            llm_backend="echo",
        )
    )


def test_ingest_then_retrieve_ranks_relevant_chunk_first(services):
    turns = services.file_transcriber.to_turns(TRANSCRIPT)
    services.ingestion.ingest_turns("m1", turns)
    results = services.retriever.retrieve("what about offline mode for Q3", "m1", {})
    assert results
    assert "offline mode" in results[0].chunk.text.lower()


def test_ingestion_is_idempotent(services):
    turns = services.file_transcriber.to_turns(TRANSCRIPT)
    services.ingestion.ingest_turns("m1", turns)
    first = services.vector_store.count()
    services.ingestion.ingest_turns("m1", turns)  # re-ingest same content
    assert services.vector_store.count() == first


def test_pii_never_reaches_the_store(services):
    turns = services.file_transcriber.to_turns(TRANSCRIPT)
    services.ingestion.ingest_turns("m1", turns)
    results = services.retriever.retrieve("how do I reach Maya", "m1", {})
    joined = " ".join(rc.chunk.text for rc in results)
    assert "maya@example.com" not in joined
    assert "[EMAIL]" in joined


def test_voice_source_normalises_to_single_speaker(services):
    res = services.ingestion.ingest_turns(
        "voice1", services.voice_transcriber.to_turns("We shipped it. Then we tested it.")
    )
    assert res["chunks"] >= 1
    assert services.vector_store.list_meetings() == ["voice1"]


def test_api_query_returns_grounded_answer_with_citations():
    services = build_services(
        Settings(embedder_backend="fake", vector_store_backend="memory",
                 reranker_backend="noop", llm_backend="echo")
    )
    turns = services.file_transcriber.to_turns(TRANSCRIPT)
    services.ingestion.ingest_turns("m1", turns)

    app.dependency_overrides[get_services] = lambda: services
    client = TestClient(app)
    try:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/query", json={"question": "what was the root cause",
                                        "meeting_id": "m1"})
        body = r.json()
        assert r.status_code == 200
        assert body["grounded"] is True
        assert len(body["citations"]) >= 1
        assert body["citations"][0]["timestamp"]
    finally:
        app.dependency_overrides.clear()


def _client_with(services):
    app.dependency_overrides[get_services] = lambda: services
    return TestClient(app)


def test_empty_ingest_is_rejected(services):
    client = _client_with(services)
    try:
        r = client.post("/ingest", json={"meeting_id": "x", "text": "no speakers here"})
        assert r.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_aggregation_query_answers_from_items(services):
    turns = services.file_transcriber.to_turns(TRANSCRIPT)
    services.ingestion.ingest_turns("m1", turns)
    client = _client_with(services)
    try:
        r = client.post("/query", json={"question": "list the action items",
                                        "meeting_id": "m1"})
        body = r.json()
        assert r.status_code == 200
        assert body["grounded"] is True
        assert "design doc" in body["text"]

        items = client.get("/items", params={"meeting_id": "m1",
                                             "kind": "action_item"}).json()["items"]
        assert any("design doc" in it["text"] for it in items)
    finally:
        app.dependency_overrides.clear()
