"""Seed the running backend with the sample transcripts.

    python scripts/seed.py            # talks to http://localhost:8000
    API_URL=... python scripts/seed.py
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

API = os.environ.get("API_URL", "http://localhost:8000")
TRANSCRIPTS = Path(__file__).resolve().parent.parent / "data" / "transcripts"


def main() -> None:
    for path in sorted(TRANSCRIPTS.glob("*.txt")):
        res = requests.post(
            f"{API}/ingest",
            json={"meeting_id": path.stem, "text": path.read_text(), "source": "file"},
            timeout=120,
        )
        res.raise_for_status()
        print(path.stem, "->", res.json())


if __name__ == "__main__":
    main()
