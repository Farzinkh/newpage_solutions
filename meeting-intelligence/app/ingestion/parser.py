"""Transcript parsing.

Primary (in-spec) line format:

    [00:01:12] Alice: Let's start with the roadmap.

Lines that don't match are treated as continuations of the previous turn, so a
speaker's multi-line paragraph stays one turn. This parser is deterministic and
has no external deps, which is why it is unit-tested directly — the fiddly,
easy-to-break logic lives here rather than behind an LLM call.

Fallback: if the timestamped parse finds nothing, we retry with an untimed
`Speaker: text` parser (a very common export format) and synthesise timestamps.
Doing this as a *whole-transcript* fallback, rather than mixing both patterns
line-by-line, keeps multi-line merging safe: a colon inside a continuation of a
properly-timestamped turn is never mistaken for a new speaker label.
"""

from __future__ import annotations

import re

from app.models import Turn

_LINE = re.compile(
    r"^\s*\[(?P<ts>[0-9:.]+)\]\s*(?P<speaker>[^:]{1,60}?):\s*(?P<text>.*)$"
)
# Untimed "Alice: text" — the speaker must look like a short name label (no
# sentence punctuation), which keeps prose lines like "Note: see doc" from being
# read as speakers.
_UNTIMED = re.compile(r"^\s*(?P<speaker>[A-Za-z][\w .'-]{0,39}?):\s+(?P<text>\S.*)$")


def _parse_timed(raw: str) -> list[Turn]:
    turns: list[Turn] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _LINE.match(line)
        if m:
            turns.append(
                Turn(
                    index=len(turns),
                    speaker=m.group("speaker").strip(),
                    timestamp=m.group("ts").strip(),
                    text=m.group("text").strip(),
                )
            )
        elif turns:
            # Continuation of the previous speaker's turn.
            turns[-1].text = f"{turns[-1].text} {line.strip()}".strip()
        # A leading non-matching line with no prior turn is dropped as preamble.
    return turns


def _parse_untimed(raw: str, seconds_per_turn: int = 15) -> list[Turn]:
    turns: list[Turn] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _UNTIMED.match(line)
        if m:
            total = len(turns) * seconds_per_turn
            ts = f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
            turns.append(
                Turn(
                    index=len(turns),
                    speaker=m.group("speaker").strip(),
                    timestamp=ts,
                    text=m.group("text").strip(),
                )
            )
        elif turns:
            turns[-1].text = f"{turns[-1].text} {line.strip()}".strip()
    return turns


def parse_transcript(raw: str) -> list[Turn]:
    turns = _parse_timed(raw)
    if turns:
        return turns
    # No in-spec lines matched — fall back to untimed "Speaker: text".
    return _parse_untimed(raw)
