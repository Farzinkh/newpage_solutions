"""Chunking.

We chunk per speaker turn rather than by fixed token windows: a turn is a
natural semantic unit in a meeting, and it lets every chunk carry the speaker
and timestamp needed for citations. Turns longer than ``max_chunk_chars`` are
split on sentence boundaries so a monologue doesn't become one giant chunk.

Chunk ids are a content hash of (meeting_id, turn_index, part, text). That is
what makes ingestion idempotent: re-ingesting the same transcript produces the
same ids, so the vector store upserts instead of duplicating.
"""

from __future__ import annotations

import hashlib
import re

from app.models import Chunk, Turn

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _chunk_id(meeting_id: str, turn_index: int, part: int, text: str) -> str:
    h = hashlib.sha256(f"{meeting_id}|{turn_index}|{part}|{text}".encode()).hexdigest()
    return h[:16]


def _hard_wrap(text: str, limit: int) -> list[str]:
    """Last-resort split for a single sentence longer than the limit — wrap on
    word boundaries so one unpunctuated monologue can't exceed the embedding
    model's token window and get silently truncated."""
    parts, current = [], ""
    for word in text.split():
        if current and len(current) + len(word) + 1 > limit:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for sentence in _SENTENCE.split(text):
        if len(sentence) > limit:
            # A single over-long sentence: flush what we have, then hard-wrap it.
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(_hard_wrap(sentence, limit))
        elif current and len(current) + len(sentence) + 1 > limit:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_turns(meeting_id: str, turns: list[Turn], max_chunk_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for turn in turns:
        for part_idx, part_text in enumerate(_split_long(turn.text, max_chunk_chars)):
            if not part_text:
                continue
            chunks.append(
                Chunk(
                    id=_chunk_id(meeting_id, turn.index, part_idx, part_text),
                    meeting_id=meeting_id,
                    speaker=turn.speaker,
                    timestamp=turn.timestamp,
                    text=part_text,
                    turn_index=turn.index,
                    occurred_at=turn.occurred_at,
                )
            )
    return chunks
