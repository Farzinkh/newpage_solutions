"""Decision & action-item extraction.

Aggregation queries — "what are all the action items?", "list the decisions",
"summarise the meeting" — are *extraction*, not retrieval: they need every
relevant turn, but top-k similarity search returns only a handful (the eval
shows a 3-turn action-item answer retrieving 1 of 3). So we tag decisions and
action items once, at ingestion, and answer those queries from the tagged
records instead.

This is a deliberately *deterministic, keyless* baseline: high-precision cue
phrases over the clean turn text. It won't catch every phrasing, but it needs no
LLM, is unit-testable, and runs offline. The documented upgrade is an LLM
extraction pass behind the same `ExtractedItem` seam — the query path that reads
these records doesn't change.
"""

from __future__ import annotations

import re

from app.models import ExtractedItem, Turn

# First-person commitments and assignments -> action items. Owner is the speaker.
# Cues are deliberately high-precision: bare "let's" / "we need to" are dropped
# because they tag agenda and filler ("let's wrap", "we need to decide") as often
# as real commitments. We favour precision over recall — a missed item is better
# than a wrong one in a list the user trusts.
_ACTION_CUES = re.compile(
    r"\b(?:i'?ll\b|i will\b|i'?m going to\b|action item|"
    r"i can (?:take|join|update|add|write|own|include|handle)\b|"
    r"follow[- ]?up|(?:can you|please) (?:own|take|handle|update)\b|"
    r"\bowns? (?:it|that|this)\b|"
    r"by (?:friday|monday|tuesday|wednesday|thursday|eod|tomorrow|next week|end of))\b",
    re.IGNORECASE,
)
# Group-level resolutions -> decisions.
_DECISION_CUES = re.compile(
    r"\b(?:decision(?:\s*[:\-])?|we (?:decided|agreed|commit|committed|will ship)|"
    r"agreed|let'?s go with|moves? to|we'?re going with|final(?:ised|ized)?)\b",
    re.IGNORECASE,
)


def _classify(text: str) -> str | None:
    # Decisions take precedence: a turn that both decides and assigns is a
    # decision at the meeting level ("Decision: ... Daniel owns it").
    if _DECISION_CUES.search(text):
        return "decision"
    if _ACTION_CUES.search(text):
        return "action_item"
    return None


def extract_items(meeting_id: str, turns: list[Turn]) -> list[ExtractedItem]:
    items: list[ExtractedItem] = []
    for turn in turns:
        kind = _classify(turn.text)
        if kind is None:
            continue
        items.append(
            ExtractedItem(
                meeting_id=meeting_id,
                kind=kind,
                speaker=turn.speaker,
                timestamp=turn.timestamp,
                turn_index=turn.index,
                text=turn.text,
            )
        )
    return items
