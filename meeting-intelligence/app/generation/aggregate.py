"""Aggregation answers from extracted items.

Some questions are enumerations, not lookups: "list the action items", "what did
we decide", "summarise the meeting". Similarity search is the wrong tool — it
returns the top-k most similar turns, not *all* the relevant ones. These are
answered directly from the decisions / action items tagged at ingestion, so the
answer is complete and every line is cited to a real turn.
"""

from __future__ import annotations

import re

from app.models import Answer, Citation, ExtractedItem

_ACTION = re.compile(r"\b(action items?|to-?dos?|tasks?|follow[- ]?ups?)\b", re.IGNORECASE)
_DECISION = re.compile(r"\b(decisions?|decided|agree(?:d|ments?)?|conclusions?)\b", re.IGNORECASE)
_SUMMARY = re.compile(r"\b(summar(?:y|ise|ize)|recap|overview|tl;?dr|key points?)\b", re.IGNORECASE)
_ENUMERATE = re.compile(r"\b(list|all|every|what are|which|show me)\b", re.IGNORECASE)


def detect_intent(question: str) -> str | None:
    """Return 'action_item', 'decision', 'summary', or None (fall through to
    normal retrieval)."""
    if _SUMMARY.search(question):
        return "summary"
    enumerating = bool(_ENUMERATE.search(question))
    if _ACTION.search(question) and enumerating:
        return "action_item"
    if _DECISION.search(question) and enumerating:
        return "decision"
    return None


def _citation(item: ExtractedItem) -> Citation:
    return Citation(
        speaker=item.speaker,
        timestamp=item.timestamp,
        chunk_id=f"{item.meeting_id}:{item.turn_index}",
        quote=item.text,
    )


def _bullets(items: list[ExtractedItem]) -> str:
    return "\n".join(
        f"- ({it.speaker} @ {it.timestamp}) {it.text}" for it in items
    )


def answer_from_items(intent: str, items: list[ExtractedItem]) -> Answer:
    if intent == "summary":
        decisions = [it for it in items if it.kind == "decision"]
        actions = [it for it in items if it.kind == "action_item"]
        if not decisions and not actions:
            return Answer(text="Not discussed in the transcript.", grounded=False)
        parts = []
        if decisions:
            parts.append("Decisions:\n" + _bullets(decisions))
        if actions:
            parts.append("Action items:\n" + _bullets(actions))
        used = decisions + actions
        return Answer(
            text="\n\n".join(parts),
            citations=[_citation(it) for it in used],
            grounded=True,
        )

    selected = [it for it in items if it.kind == intent]
    if not selected:
        label = "action items" if intent == "action_item" else "decisions"
        return Answer(text=f"No {label} were recorded in the transcript.", grounded=False)
    heading = "Action items:" if intent == "action_item" else "Decisions:"
    return Answer(
        text=f"{heading}\n{_bullets(selected)}",
        citations=[_citation(it) for it in selected],
        grounded=True,
    )
