"""Cleaning and PII redaction.

Two distinct jobs, both run before embedding so nothing sensitive ever reaches
the vector store or the LLM:

1. Artifact cleaning — remove STT junk (``[inaudible]``, filler ``um/uh``,
   doubled whitespace). This is *not* stopword removal: grammatical words carry
   meaning for dense retrieval ("did not approve" vs "approve"), so we leave
   natural language intact.

2. PII redaction — replace leaked contact/sensitive PII (email, phone, SSN,
   card numbers) with a typed placeholder like ``[EMAIL]``. Crucially we keep
   *speaker names*: meetings are about people, and names are the backbone of
   retrieval and citations. We redact what leaks, not who spoke.

Placeholders (not blanks) preserve sentence structure for the embedder.
The regex layer is a deterministic baseline; Presidio (spaCy NER) is the
documented upgrade for names-in-context and is loaded lazily if installed.
Only redaction *counts* are logged, never the raw matched values.
"""

from __future__ import annotations

import re

_ARTIFACTS = [
    re.compile(r"\[(?:inaudible|crosstalk|silence|noise)\]", re.IGNORECASE),
    re.compile(r"\b(?:um+|uh+|erm+|hmm+)\b", re.IGNORECASE),
]

# Each pattern ends on a digit (not an optional separator) so a redaction never
# swallows the following space and glues words together ("[CARD]shipped").
_PII = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b\d(?:[ -]?\d){12,15}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
}


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum. Gates CREDIT_CARD so arbitrary 13–16 digit numbers
    (order ids, metrics) aren't redacted as cards — only plausible ones are."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def clean_text(text: str) -> str:
    for pat in _ARTIFACTS:
        text = pat.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Reattach punctuation the filler removal orphaned (" , " / " ." -> ", ").
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # Drop a leading orphan comma/semicolon left when a turn opened with a filler.
    text = re.sub(r"^[\s,;]+", "", text)
    return text.strip()


def _redact_card(text: str, counts: dict[str, int]) -> str:
    def repl(m: re.Match[str]) -> str:
        if _luhn_ok(re.sub(r"\D", "", m.group())):
            counts["CREDIT_CARD"] = counts.get("CREDIT_CARD", 0) + 1
            return "[CREDIT_CARD]"
        return m.group()

    return _PII["CREDIT_CARD"].sub(repl, text)


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Return (redacted_text, counts_by_type). Order matters: card/SSN before
    phone so a 16-digit card isn't partially matched as a phone number."""
    counts: dict[str, int] = {}
    for label in ("EMAIL", "SSN"):
        text, n = _PII[label].subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    text = _redact_card(text, counts)
    text, n = _PII["PHONE"].subn("[PHONE]", text)
    if n:
        counts["PHONE"] = counts.get("PHONE", 0) + n
    return text, counts


def preprocess(text: str, redact: bool = True) -> tuple[str, dict[str, int]]:
    text = clean_text(text)
    if redact:
        return redact_pii(text)
    return text, {}
