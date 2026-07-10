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
    timestamp: str  # human-readable, e.g. "00:01:12"; kept as text for citations
    text: str


class Chunk(BaseModel):
    """A retrievable unit. One chunk per turn by default (see chunker)."""

    id: str
    meeting_id: str
    speaker: str
    timestamp: str
    text: str
    turn_index: int

    def metadata(self) -> dict[str, str | int]:
        return {
            "meeting_id": self.meeting_id,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "turn_index": self.turn_index,
        }


class RetrievedChunk(BaseModel):
    """A chunk plus the scores that got it here. Both scores are surfaced to the
    UI and logs so retrieval quality is inspectable, not a black box."""

    chunk: Chunk
    similarity: float
    rerank_score: float | None = None


class Citation(BaseModel):
    speaker: str
    timestamp: str
    chunk_id: str
    quote: str


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    # Debug view of what retrieval returned for this question.
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    grounded: bool = True
