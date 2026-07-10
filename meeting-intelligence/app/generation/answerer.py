"""Answer generation with guardrails.

The guardrail is two-layered:

1. Prompt-level: the model is instructed to answer *only* from the numbered
   context, cite sources as ``[n]``, and say "Not discussed in the transcript."
   when the answer is absent.

2. Output-level: we parse the ``[n]`` markers, drop any that point outside the
   retrieved set, and map the valid ones back to real (speaker, timestamp,
   quote) citations. If the model answered but cited nothing valid, we flag the
   answer as ungrounded rather than presenting it as sourced.
"""

from __future__ import annotations

import logging
import re

from app.generation.conversation import format_history
from app.generation.llm import LLMClient
from app.logging_config import log_event, timed
from app.models import Answer, Citation, HistoryTurn, RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a meeting intelligence assistant. Answer the user's question using "
    "ONLY the numbered transcript excerpts provided. Every claim must cite its "
    "source as [n] using the excerpt number. If the excerpts do not contain the "
    "answer, reply exactly: 'Not discussed in the transcript.' Do not use outside "
    "knowledge and do not invent speakers, dates, or figures. The conversation so "
    "far is provided only to resolve references in the question; never cite it."
)

_CITE = re.compile(r"\[(\d+)\]")
_NOT_DISCUSSED = "not discussed in the transcript"


def _build_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        lines.append(f"[{i}] ({c.speaker} @ {c.timestamp}) {c.text}")
    return "\n".join(lines)


class Answerer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        timings: dict[str, float],
        history: list[HistoryTurn] | None = None,
        brief: str | None = None,
    ) -> Answer:
        if not chunks:
            return Answer(text="Not discussed in the transcript.", grounded=False)

        context = _build_context(chunks)
        convo = f"Conversation so far:\n{format_history(history)}\n\n" if history else ""
        # Whole-meeting brief as orientation, clearly marked as non-citable so
        # the model grounds specific claims in the numbered excerpts, not here.
        overview = (
            f"Meeting overview (background for context — do NOT cite this, cite the "
            f"numbered excerpts):\n{brief}\n\n" if brief else ""
        )
        user = (
            f"{convo}{overview}Transcript excerpts:\n{context}\n\nQuestion: {question}"
        )

        with timed(timings, "llm_ms"):
            raw = self._llm.complete(SYSTEM_PROMPT, user).strip()

        if raw.lower().startswith(_NOT_DISCUSSED):
            return Answer(text=raw, retrieved=chunks, grounded=False)

        # Output-level guardrail: keep only in-range citations.
        cited_idxs = {
            int(n) for n in _CITE.findall(raw) if 1 <= int(n) <= len(chunks)
        }
        citations = [
            Citation(
                speaker=chunks[i - 1].chunk.speaker,
                timestamp=chunks[i - 1].chunk.timestamp,
                chunk_id=chunks[i - 1].chunk.id,
                quote=chunks[i - 1].chunk.text,
            )
            for i in sorted(cited_idxs)
        ]
        grounded = bool(citations)
        if not grounded:
            log_event(logger, "ungrounded_answer", question=question)

        return Answer(text=raw, citations=citations, retrieved=chunks, grounded=grounded)
