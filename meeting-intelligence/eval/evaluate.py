"""Retrieval evaluation harness.

Runs each gold question through ingestion + retrieval and reports hit-rate@k and
Mean Reciprocal Rank. This is what turns every retrieval knob (top_k, rerank
on/off, embedding model) from a guess into a measured decision — run it before
and after a change and compare.

Usage:
    python -m eval.evaluate                # uses whatever backends config selects
    EMBEDDER_BACKEND=fake VECTOR_STORE_BACKEND=memory python -m eval.evaluate

Note: with the fake embedder the numbers are a plumbing check, not a real
quality measure. Point it at the local (or hosted) embedder for meaningful
scores.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.factory import build_services

ROOT = Path(__file__).resolve().parent.parent
GOLD = Path(__file__).resolve().parent / "gold_set.json"
TRANSCRIPTS = ROOT / "data" / "transcripts"


def _ingest_all(services) -> None:
    for path in sorted(TRANSCRIPTS.glob("*.txt")):
        turns = services.file_transcriber.to_turns(path.read_text())
        services.ingestion.ingest_turns(path.stem, turns)


def evaluate() -> dict[str, float]:
    settings = Settings()
    services = build_services(settings)
    _ingest_all(services)

    cases = json.loads(GOLD.read_text())["cases"]
    k = settings.final_top_k
    hits, reciprocal_ranks = 0, []

    for case in cases:
        timings: dict[str, float] = {}
        results = services.retriever.retrieve(
            case["question"], case["meeting_id"], timings
        )
        got = [rc.chunk.turn_index for rc in results]
        relevant = set(case["relevant_turn_indices"])

        hit = any(idx in relevant for idx in got)
        hits += int(hit)
        rr = 0.0
        for rank, idx in enumerate(got, start=1):
            if idx in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        status = "HIT " if hit else "MISS"
        print(f"[{status}] {case['meeting_id']}: {case['question']}")
        print(f"         expected turns {sorted(relevant)}, got {got}")

    n = len(cases)
    metrics = {
        f"hit_rate@{k}": round(hits / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "n_cases": n,
    }
    print("\n=== metrics ===")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
