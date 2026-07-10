"""Two-stage cross-meeting retrieval: pick the meetings that matter, inject each
one's brief, and dig each hit with its before/after context."""

from app.config import Settings
from app.factory import build_services
from app.interfaces import LLMClient
from app.retrieval.planner import plan_blocks

M1 = (
    "[00:00:04] Priya: We commit offline mode for Q3, single device only.\n"
    "[00:00:19] Priya: The sync conflict resolution is the risky part.\n"
    "[00:00:34] Priya: So we timebox it to four weeks.\n"
)
M2 = (
    "[00:00:05] Daniel: Root cause of the outage was a low connection pool limit.\n"
    "[00:00:20] Daniel: We raised the pool ceiling and added an alert.\n"
)


class _CapturingLLM(LLMClient):
    def __init__(self) -> None:
        self.last_user = ""

    def complete(self, system: str, user: str) -> str:
        self.last_user = user
        return "Offline mode is risky due to sync conflicts. [2]"


def _services(llm=None):
    s = build_services(Settings(embedder_backend="fake", vector_store_backend="memory",
                                reranker_backend="noop", llm_backend="echo",
                                context_window=1, max_meetings=3, per_meeting_hits=1))
    if llm is not None:
        s.llm = llm
        s.answerer._llm = llm
    s.ingestion.ingest_turns("planning", s.file_transcriber.to_turns(M1))
    s.ingestion.ingest_turns("incident", s.file_transcriber.to_turns(M2))
    return s


def test_planner_expands_hits_with_neighbours():
    s = _services()
    cands = s.retriever.candidates("sync conflict resolution risky", "planning", {})
    blocks = plan_blocks(s.vector_store, s.item_store, cands, s.settings, "planning")
    assert len(blocks) == 1
    presented = blocks[0].presented
    hit_turns = [p.chunk.turn_index for p in presented if p.is_hit]
    all_turns = [p.chunk.turn_index for p in presented]
    # The hit on turn 1 pulls in its neighbours (turns 0 and 2) as context.
    assert 1 in hit_turns
    assert 0 in all_turns and 2 in all_turns
    # Context turns are marked as not-hits.
    assert any(not p.is_hit for p in presented)


def test_context_turns_are_ordered_chronologically():
    s = _services()
    cands = s.retriever.candidates("timebox four weeks", "planning", {})
    blocks = plan_blocks(s.vector_store, s.item_store, cands, s.settings, "planning")
    turns = [p.chunk.turn_index for p in blocks[0].presented]
    assert turns == sorted(turns)


def test_cross_meeting_query_selects_multiple_meetings():
    s = _services()
    cands = s.retriever.candidates("offline mode risk and outage root cause", None, {})
    blocks = plan_blocks(s.vector_store, s.item_store, cands, s.settings, None)
    ids = {b.meeting_id for b in blocks}
    assert ids == {"planning", "incident"}
    # Every block carries its own brief for orientation.
    assert all(b.brief_text for b in blocks)


def test_grouped_prompt_labels_each_meeting_and_context():
    llm = _CapturingLLM()
    s = _services(llm)
    s.query("What are the risks discussed and what caused the outage?")
    prompt = llm.last_user
    assert "=== Meeting: planning ===" in prompt
    assert "=== Meeting: incident ===" in prompt
    assert "(context)" in prompt  # neighbour turns are marked
