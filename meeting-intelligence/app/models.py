"""Domain models.

These are deliberately plain and framework-agnostic: the pipeline speaks in
`Turn` / `Chunk` / `RetrievedChunk` regardless of which embedder, store, or LLM
is wired in behind the interfaces. Keeping the domain types free of any vendor
detail is what lets the seams in `interfaces.py` stay swappable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """A single speaker turn, the atomic unit both file and voice inputs
    normalise to before anything else runs."""

    index: int
    speaker: str
    timestamp: str  # relative to meeting start, e.g. "00:01:12"
    text: str
    # Absolute wall-clock time (meeting_start + offset), set when the meeting's
    # start datetime is known. This is what disambiguates turns across meetings.
    occurred_at: str | None = None


class Chunk(BaseModel):
    """A retrievable unit. One chunk per turn by default (see chunker)."""

    id: str
    meeting_id: str
    speaker: str
    timestamp: str
    text: str
    turn_index: int
    occurred_at: str | None = None

    def display_time(self) -> str:
        """Absolute time when we have it (unambiguous across meetings), else the
        relative transcript timestamp."""
        return self.occurred_at or self.timestamp

    def metadata(self) -> dict[str, str | int]:
        meta: dict[str, str | int] = {
            "meeting_id": self.meeting_id,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "turn_index": self.turn_index,
        }
        if self.occurred_at:
            meta["occurred_at"] = self.occurred_at
        return meta


class RetrievedChunk(BaseModel):
    """A chunk plus the scores that got it here. Both scores are surfaced to the
    UI and logs so retrieval quality is inspectable, not a black box."""

    chunk: Chunk
    similarity: float
    rerank_score: float | None = None


class ExtractedItem(BaseModel):
    """A decision or action item pulled out at ingestion by the deterministic
    extractor, so aggregation queries ("list the action items") are answered
    from structured records instead of top-k similarity search."""

    meeting_id: str
    kind: str  # "action_item" | "decision"
    speaker: str
    timestamp: str
    turn_index: int
    text: str
    occurred_at: str | None = None

    def display_time(self) -> str:
        return self.occurred_at or self.timestamp


class MeetingBrief(BaseModel):
    """Meeting-level highlights derived once at ingestion. Injected as background
    at query time so an answer sees the whole meeting's shape (participants, what
    was decided, who owns what), not just the isolated top-k retrieved turns —
    the classic local-vs-global gap in chunk retrieval."""

    meeting_id: str
    participants: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)

    def render(self, max_items: int = 6, max_chars: int = 200) -> str:
        def _lines(items: list[str]) -> str:
            return "\n".join(f"  - {t[:max_chars]}" for t in items[:max_items]) or "  - (none)"

        return (
            f"Participants: {', '.join(self.participants) or '(unknown)'}\n"
            f"Decisions:\n{_lines(self.decisions)}\n"
            f"Action items:\n{_lines(self.action_items)}"
        )


class HistoryTurn(BaseModel):
    """One prior message in the conversation, fed back so follow-up questions
    ("who owns that?") resolve against what was already said."""

    role: str  # "user" | "assistant"
    content: str


class Citation(BaseModel):
    meeting_id: str
    speaker: str
    timestamp: str  # absolute datetime when known, else relative to meeting start
    chunk_id: str
    quote: str


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    # Debug view of what retrieval returned for this question.
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    grounded: bool = True
