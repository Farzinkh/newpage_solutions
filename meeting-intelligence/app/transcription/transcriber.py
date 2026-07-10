"""Transcription / input normalisation.

The `Transcriber` seam is where the two input sources are reconciled. A file
already carries speaker + timestamp structure, so `FileTranscriber` just parses
it. Browser voice-to-text returns a flat string with no diarisation, so
`PlainTextTranscriber` wraps it as a single speaker with synthesised timestamps.

Diarisation (telling participants apart) needs the raw waveform and a
speaker-embedding model (e.g. pyannote: VAD -> x-vectors -> clustering), which
is impractical client-side. The intended upgrade is a server-side backend
(Deepgram / AssemblyAI / Whisper+pyannote) that returns diarised turns directly
and slots in here as a new `Transcriber` — the rest of the system is unchanged.
"""

from __future__ import annotations

import re

from app.ingestion.parser import parse_transcript
from app.interfaces import Transcriber
from app.models import Turn

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class FileTranscriber(Transcriber):
    """Structured transcript text -> turns."""

    def to_turns(self, raw_text: str) -> list[Turn]:
        return parse_transcript(raw_text)


class PlainTextTranscriber(Transcriber):
    """Unstructured voice-to-text -> single-speaker turns.

    No diarisation available from the browser, so everything is attributed to
    one speaker. Timestamps are synthesised (one tick per sentence) purely so
    citations have *a* anchor; they are approximate by construction, which the
    UI/README flags as a known limitation of the voice path.
    """

    def __init__(self, speaker: str = "Speaker 1", seconds_per_turn: int = 15) -> None:
        self._speaker = speaker
        self._step = seconds_per_turn

    def to_turns(self, raw_text: str) -> list[Turn]:
        sentences = [s.strip() for s in _SENTENCE.split(raw_text) if s.strip()]
        turns: list[Turn] = []
        for i, sentence in enumerate(sentences):
            total = i * self._step
            ts = f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
            turns.append(Turn(index=i, speaker=self._speaker, timestamp=ts, text=sentence))
        return turns
