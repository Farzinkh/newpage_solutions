"""The meeting brief (highlights) gives an answer whole-meeting context, not just
the isolated top-k chunks. These tests prove the brief is built at ingestion and
actually reaches the LLM prompt."""

from app.config import Settings
from app.factory import build_services
from app.interfaces import LLMClient

TRANSCRIPT = (
    "[00:00:04] Priya: Decision: offline mode ships in Q3, single-device only.\n"
    "[00:00:20] Daniel: I'll write the design doc by Friday.\n"
    "[00:00:35] Maya: Sounds good to me.\n"
)


class _CapturingLLM(LLMClient):
    """Records the user prompt so we can assert what context the model saw."""

    def __init__(self) -> None:
        self.last_user = ""

    def complete(self, system: str, user: str) -> str:
        self.last_user = user
        return "Offline mode ships in Q3. [1]"


def _services_with(llm):
    s = build_services(
        Settings(embedder_backend="fake", vector_store_backend="memory",
                 reranker_backend="noop", llm_backend="echo")
    )
    s.llm = llm
    s.answerer._llm = llm  # swap in the capturing client
    return s


def test_brief_built_at_ingestion():
    s = _services_with(_CapturingLLM())
    s.ingestion.ingest_turns("m1", s.file_transcriber.to_turns(TRANSCRIPT))
    brief = s.item_store.get_brief("m1")
    assert brief is not None
    assert brief.participants == ["Priya", "Daniel", "Maya"]  # first-seen order
    assert any("offline mode" in d.lower() for d in brief.decisions)
    assert any("design doc" in a.lower() for a in brief.action_items)


def test_brief_is_injected_into_the_answer_prompt():
    llm = _CapturingLLM()
    s = _services_with(llm)
    s.ingestion.ingest_turns("m1", s.file_transcriber.to_turns(TRANSCRIPT))
    s.query("What is the risk on offline mode?", "m1")
    assert "Meeting overview" in llm.last_user
    assert "Participants: Priya, Daniel, Maya" in llm.last_user
