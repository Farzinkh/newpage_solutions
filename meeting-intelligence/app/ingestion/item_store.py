"""Per-meeting derived records: extracted decisions / action items, and the
meeting brief (highlights).

Kept separate from the vector store because these are used differently from
chunks: items are answered by *enumeration* (not similarity search), and the
brief is injected as whole-meeting context. The in-memory implementation matches
the project's "start simple" posture; the production version is a table keyed by
meeting_id (the same pgvector/RDS instance that backs the vector store), a new
implementation of this same tiny interface.

Idempotent per meeting: re-ingesting replaces that meeting's records.
"""

from __future__ import annotations

from app.models import ExtractedItem, MeetingBrief


class InMemoryItemStore:
    def __init__(self) -> None:
        self._items: dict[str, list[ExtractedItem]] = {}
        self._briefs: dict[str, MeetingBrief] = {}

    def replace(self, meeting_id: str, items: list[ExtractedItem]) -> None:
        self._items[meeting_id] = items

    def list(self, meeting_id: str | None = None, kind: str | None = None) -> list[ExtractedItem]:
        if meeting_id is not None:
            pool = self._items.get(meeting_id, [])
        else:
            pool = [item for items in self._items.values() for item in items]
        if kind is not None:
            pool = [item for item in pool if item.kind == kind]
        return pool

    def set_brief(self, meeting_id: str, brief: MeetingBrief) -> None:
        self._briefs[meeting_id] = brief

    def get_brief(self, meeting_id: str | None) -> MeetingBrief | None:
        return self._briefs.get(meeting_id) if meeting_id else None
