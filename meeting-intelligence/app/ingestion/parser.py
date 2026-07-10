"""Transcript parsing.

Expected line format:

    [00:01:12] Alice: Let's start with the roadmap.

Lines that don't match are treated as continuations of the previous turn, so a
speaker's multi-line paragraph stays one turn. This parser is deterministic and
has no external deps, which is why it is unit-tested directly — the fiddly,
easy-to-break logic lives here rather than behind an LLM call.
"""

from __future__ import annotations

import re

from app.models import Turn

_LINE = re.compile(
    r"^\s*\[(?P<ts>[0-9:.]+)\]\s*(?P<speaker>[^:]{1,60}?):\s*(?P<text>.*)$"
)


def parse_transcript(raw: str) -> list[Turn]:
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
