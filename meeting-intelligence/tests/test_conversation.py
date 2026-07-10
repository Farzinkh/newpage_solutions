from app.generation.conversation import format_history, recent, rewrite_query
from app.generation.llm import EchoLLM
from app.interfaces import LLMClient
from app.models import HistoryTurn


class _RewriteLLM(LLMClient):
    """Stand-in for a real generative LLM: echoes a fixed standalone question."""

    def complete(self, system: str, user: str) -> str:
        return "Which candidate does Daniel prefer?"


def _history():
    return [
        HistoryTurn(role="user", content="Who is advancing to onsite?"),
        HistoryTurn(role="assistant", content="Priya is advancing to onsite."),
    ]


def test_recent_caps_history():
    hist = [HistoryTurn(role="user", content=str(i)) for i in range(10)]
    assert len(recent(hist, 4)) == 4
    assert recent(hist, 0) == []


def test_rewrite_uses_llm_when_available():
    out = rewrite_query(_RewriteLLM(), _history(), "What about Daniel?")
    assert out == "Which candidate does Daniel prefer?"


def test_rewrite_falls_back_without_generation():
    # EchoLLM can't generate, so the fallback carries the prior question's terms
    # into the retrieval query instead of feeding Echo a prompt it would mangle.
    out = rewrite_query(EchoLLM(), _history(), "What about him?")
    assert "onsite" in out and "What about him?" in out


def test_no_history_returns_question_unchanged():
    assert rewrite_query(EchoLLM(), [], "Who owns the design doc?") == (
        "Who owns the design doc?"
    )


def test_format_history_renders_roles():
    assert "User:" in format_history(_history())
    assert "Assistant:" in format_history(_history())
