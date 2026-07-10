"""Seed the running backend with the sample transcripts.

    python scripts/seed.py            # talks to http://localhost:8000
    API_URL=... python scripts/seed.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ingestion.timestamps import parse_filename  # noqa: E402

API = os.environ.get("API_URL", "http://localhost:8000")
TRANSCRIPTS = Path(__file__).resolve().parent.parent / "data" / "transcripts"


def main() -> None:
    for path in sorted(TRANSCRIPTS.glob("*.txt")):
        meeting_id, started_at = parse_filename(path.name)
        payload = {"meeting_id": meeting_id, "text": path.read_text(), "source": "file"}
        if started_at is not None:
            payload["started_at"] = started_at.isoformat()
        res = requests.post(f"{API}/ingest", json=payload, timeout=120)
        res.raise_for_status()
        print(meeting_id, "->", res.json())


if __name__ == "__main__":
    main()
