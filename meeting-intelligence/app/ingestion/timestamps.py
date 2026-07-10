"""Absolute-time preprocessing.

Transcript timestamps are *relative* to the meeting start (``00:02:36``). That's
fine within one meeting but ambiguous the moment answers span several — the same
``00:02:36`` occurs in every meeting. So we anchor each turn to a real wall-clock
datetime: ``meeting_start + offset``.

The meeting start comes from the file name (the convention below) or is passed
explicitly at ingest. If neither is available we simply keep the relative string,
so this is a strictly additive enhancement — nothing breaks without a date.

File-name convention (date required, time optional; separators flexible):
    2026-06-03_1548_product_planning.txt  -> ("product_planning", 2026-06-03 15:48)
    product_planning_20260603.txt         -> ("product_planning", 2026-06-03 00:00)
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_DATE = re.compile(r"(?P<y>\d{4})[-_]?(?P<mo>\d{2})[-_]?(?P<d>\d{2})")
_TIME = re.compile(r"(?<!\d)(?P<h>\d{2})[-:]?(?P<mi>\d{2})(?:[-:]?(?P<s>\d{2}))?(?!\d)")


def parse_offset(ts: str) -> timedelta | None:
    """Parse a relative transcript timestamp into a duration since meeting start.
    Accepts ``HH:MM:SS(.ms)`` or ``MM:SS``; returns None if unparseable."""
    parts = ts.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, m, s = 0.0, nums[0], nums[1]
    else:
        return None
    return timedelta(hours=h, minutes=m, seconds=s)


def to_absolute(started_at: datetime, ts: str) -> str | None:
    """meeting_start + relative offset -> 'YYYY-MM-DD HH:MM:SS'."""
    offset = parse_offset(ts)
    if offset is None:
        return None
    return (started_at + offset).strftime("%Y-%m-%d %H:%M:%S")


def parse_filename(name: str) -> tuple[str, datetime | None]:
    """Return (meeting_id, meeting_start). The meeting_id is the file stem with
    the date/time tokens and stray separators stripped; meeting_start is None if
    no date token is present."""
    stem = re.sub(r"\.[^.]+$", "", name)  # drop extension

    started_at: datetime | None = None
    dm = _DATE.search(stem)
    if dm:
        rest = stem[dm.end():]
        tm = _TIME.search(rest)
        h = int(tm.group("h")) if tm else 0
        mi = int(tm.group("mi")) if tm else 0
        s = int(tm.group("s")) if tm and tm.group("s") else 0
        try:
            started_at = datetime(
                int(dm.group("y")), int(dm.group("mo")), int(dm.group("d")), h, mi, s
            )
        except ValueError:
            started_at = None  # e.g. an id that merely looks like a date

    # meeting_id: remove the matched date (and following time) then tidy separators
    meeting_id = stem
    if started_at is not None:
        meeting_id = (_DATE.sub("", stem, count=1))
        meeting_id = _TIME.sub("", meeting_id, count=1)
    meeting_id = re.sub(r"[-_\s]{2,}", "_", meeting_id).strip("-_ ")
    return meeting_id or stem, started_at
