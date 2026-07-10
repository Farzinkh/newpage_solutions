"""Two-stage, cross-meeting retrieval planning.

Plain top-k throws a flat list of chunks at the model. For a question that
touches several meetings that loses two things the model needs: *which meeting
each turn came from* and *the turns around each hit*. This planner rebuilds that
structure:

  Stage 1 (coarse — which meetings): group the wide-net candidates by meeting,
  rank meetings by their best hit, and keep the top few. Each selected meeting
  contributes its brief (highlights) as orientation.

  Stage 2 (fine — dig in): for each selected meeting take its top hits and expand
  each with `context_window` turns before and after (a metadata lookup, not a
  second similarity search), so the model reads the hit *in the flow of the
  conversation*.

The output is a list of per-meeting blocks; the answerer renders them grouped and
numbers every shown turn so any of them — hit or context — can be cited.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.ingestion.item_store import InMemoryItemStore
from app.interfaces import VectorStore
from app.models import Chunk, RetrievedChunk


@dataclass
class Presented:
    chunk: Chunk
    is_hit: bool  # a similarity hit vs. an expanded-context neighbour
    similarity: float | None


@dataclass
class MeetingBlock:
    meeting_id: str
    brief_text: str | None
    presented: list[Presented]


def _select_meetings(
    by_meeting: dict[str, list[RetrievedChunk]],
    scope_meeting_id: str | None,
    max_meetings: int,
) -> list[str]:
    if scope_meeting_id is not None:
        return [scope_meeting_id] if scope_meeting_id in by_meeting else []
    # Candidates arrive sorted by similarity, so each list's first item is that
    # meeting's best hit. Rank meetings by that and keep the strongest few.
    ranked = sorted(by_meeting.items(), key=lambda kv: kv[1][0].similarity, reverse=True)
    return [mid for mid, _ in ranked[:max_meetings]]


def plan_blocks(
    store: VectorStore,
    item_store: InMemoryItemStore,
    candidates: list[RetrievedChunk],
    settings: Settings,
    scope_meeting_id: str | None,
) -> list[MeetingBlock]:
    by_meeting: dict[str, list[RetrievedChunk]] = {}
    for rc in candidates:
        by_meeting.setdefault(rc.chunk.meeting_id, []).append(rc)

    blocks: list[MeetingBlock] = []
    for mid in _select_meetings(by_meeting, scope_meeting_id, settings.max_meetings):
        hits = by_meeting[mid][: settings.per_meeting_hits]
        hit_sim = {h.chunk.id: h.similarity for h in hits}
        hit_ids = set(hit_sim)

        # Expand each hit with its neighbouring turns.
        want: set[int] = set()
        for h in hits:
            ti = h.chunk.turn_index
            want.update(i for i in range(ti - settings.context_window,
                                         ti + settings.context_window + 1) if i >= 0)
        neighbours = store.fetch_turns(mid, sorted(want))

        seen: set[str] = set()
        presented: list[Presented] = []
        for c in sorted(neighbours, key=lambda c: (c.turn_index, c.id)):
            if c.id in seen:
                continue
            seen.add(c.id)
            presented.append(Presented(c, c.id in hit_ids, hit_sim.get(c.id)))

        brief = item_store.get_brief(mid)
        blocks.append(MeetingBlock(mid, brief.render() if brief else None, presented))
    return blocks


def hits_of(blocks: list[MeetingBlock]) -> list[RetrievedChunk]:
    """The similarity hits across all blocks, for the UI score view and logs."""
    return [
        RetrievedChunk(chunk=p.chunk, similarity=p.similarity or 0.0)
        for b in blocks
        for p in b.presented
        if p.is_hit
    ]
