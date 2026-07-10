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
from app.retrieval.planner import MeetingBlock, hits_of

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a meeting intelligence assistant. Answer the user's question using "
    "ONLY the numbered transcript excerpts provided. Every claim must cite its "
    "source as [n] using the excerpt number. If the excerpts do not contain the "
    "answer, reply exactly: 'Not discussed in the transcript.' Do not use outside "
    "knowledge and do not invent speakers, dates, or figures. The conversation so "
    "far is provided only to resolve references in the question; never cite it."
)

SYSTEM_PROMPT_MULTI = (
    "You are a meeting intelligence assistant. You are given transcript excerpts "
    "grouped by meeting, each meeting prefixed with a short overview for "
    "orientation. Answer using ONLY the numbered excerpts, and cite every claim "
    "as [n]. Excerpts marked '(context)' are the turns surrounding a hit — use "
    "them to read a quote in context. When the answer draws on more than one "
    "meeting, attribute each part to its meeting by name. Use the overviews only "
    "for orientation — never cite them. If the excerpts do not contain the "
    "answer, reply exactly: 'Not discussed in the transcript.' Do not use outside "
    "knowledge or invent speakers, dates, or figures. Any conversation history is "
    "only to resolve references in the question; never cite it."
)

_FOCUS_SYSTEM = (
    "You are planning how to answer a question about several meetings. Given the "
    "question and each meeting's overview, state in 1-2 sentences which meeting(s) "
    "most likely hold the answer and what to look for. This is a hypothesis to "
    "verify against the excerpts, not the answer."
)

_CITE = re.compile(r"\[(\d+)\]")
_NOT_DISCUSSED = "not discussed in the transcript"


def _build_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        # Label the meeting and absolute time so the model can attribute and
        # order turns correctly when excerpts span several meetings.
        lines.append(f"[{i}] ({c.meeting_id} · {c.speaker} @ {c.display_time()}) {c.text}")
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
                meeting_id=chunks[i - 1].chunk.meeting_id,
                speaker=chunks[i - 1].chunk.speaker,
                timestamp=chunks[i - 1].chunk.display_time(),
                chunk_id=chunks[i - 1].chunk.id,
                quote=chunks[i - 1].chunk.text,
            )
            for i in sorted(cited_idxs)
        ]
        grounded = bool(citations)
        if not grounded:
            log_event(logger, "ungrounded_answer", question=question)

        return Answer(text=raw, citations=citations, retrieved=chunks, grounded=grounded)

    def early_focus(self, question: str, blocks: list[MeetingBlock]) -> str | None:
        """First pass: read the meeting overviews and form a hypothesis of which
        meeting matters and what to look for. Skipped for a non-generative LLM."""
        if not self._llm.can_generate or len(blocks) < 2:
            return None
        overviews = "\n\n".join(
            f"Meeting: {b.meeting_id}\n{b.brief_text or '(no overview)'}" for b in blocks
        )
        note = self._llm.complete(_FOCUS_SYSTEM, f"Question: {question}\n\n{overviews}")
        return note.strip() or None

    def answer_grouped(
        self,
        question: str,
        blocks: list[MeetingBlock],
        timings: dict[str, float],
        history: list[HistoryTurn] | None = None,
        focus: str | None = None,
    ) -> Answer:
        presented = [p for b in blocks for p in b.presented]
        if not presented:
            return Answer(text="Not discussed in the transcript.", grounded=False)

        # Render grouped-by-meeting context, numbering every shown turn.
        lines: list[str] = []
        n = 0
        for b in blocks:
            lines.append(f"=== Meeting: {b.meeting_id} ===")
            if b.brief_text:
                lines.append(f"Overview:\n{b.brief_text}")
            lines.append("Excerpts (chronological):")
            for p in b.presented:
                n += 1
                c = p.chunk
                marker = "" if p.is_hit else " (context)"
                lines.append(
                    f"[{n}] ({c.meeting_id} · {c.speaker} @ {c.display_time()}){marker} {c.text}"
                )
        context = "\n".join(lines)

        convo = f"Conversation so far:\n{format_history(history)}\n\n" if history else ""
        hypo = f"Working hypothesis (verify against the excerpts):\n{focus}\n\n" if focus else ""
        user = f"{convo}{hypo}{context}\n\nQuestion: {question}"

        with timed(timings, "llm_ms"):
            raw = self._llm.complete(SYSTEM_PROMPT_MULTI, user).strip()

        retrieved = hits_of(blocks)
        if raw.lower().startswith(_NOT_DISCUSSED):
            return Answer(text=raw, retrieved=retrieved, grounded=False)

        cited_idxs = {int(x) for x in _CITE.findall(raw) if 1 <= int(x) <= len(presented)}
        citations = [
            Citation(
                meeting_id=presented[i - 1].chunk.meeting_id,
                speaker=presented[i - 1].chunk.speaker,
                timestamp=presented[i - 1].chunk.display_time(),
                chunk_id=presented[i - 1].chunk.id,
                quote=presented[i - 1].chunk.text,
            )
            for i in sorted(cited_idxs)
        ]
        grounded = bool(citations)
        if not grounded:
            log_event(logger, "ungrounded_answer", question=question)
        return Answer(text=raw, citations=citations, retrieved=retrieved, grounded=grounded)
