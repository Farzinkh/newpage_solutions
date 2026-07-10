"""Conversation handling: multi-turn memory and follow-up resolution.

Two jobs, both about making the assistant *conversational* rather than a
stateless Q&A box:

1. `rewrite_query` turns a context-dependent follow-up ("who owns that?") into a
   standalone question before retrieval, so the embedding query carries the
   referent. This needs real generation, so it is skipped for the extractive
   EchoLLM (`can_generate is False`); a cheap keyless fallback prepends the last
   user question so at least its terms reach the retriever.

2. `format_history` renders the recent turns for the answer prompt, so the model
   can resolve references and avoid repeating itself.
"""

from __future__ import annotations

from app.interfaces import LLMClient
from app.models import HistoryTurn

_REWRITE_SYSTEM = (
    "Rewrite the user's latest question into a single standalone question that "
    "makes sense without the conversation history. Resolve pronouns and "
    "references using the history. Output ONLY the rewritten question, nothing "
    "else. If it is already standalone, return it unchanged."
)


def recent(history: list[HistoryTurn], max_turns: int) -> list[HistoryTurn]:
    return history[-max_turns:] if max_turns > 0 else []


def format_history(history: list[HistoryTurn]) -> str:
    lines = [f"{h.role.capitalize()}: {h.content}" for h in history]
    return "\n".join(lines)


def rewrite_query(llm: LLMClient, history: list[HistoryTurn], question: str) -> str:
    """Return a standalone version of `question` for retrieval."""
    if not history:
        return question
    if not llm.can_generate:
        # Keyless fallback: carry the previous user question's terms so a bare
        # follow-up still retrieves something relevant.
        prev_user = next(
            (h.content for h in reversed(history) if h.role == "user"), ""
        )
        return f"{prev_user} {question}".strip()

    convo = format_history(history)
    user = f"Conversation history:\n{convo}\n\nLatest question: {question}"
    rewritten = llm.complete(_REWRITE_SYSTEM, user).strip()
    return rewritten or question
