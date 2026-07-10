"""Retrieval orchestration: embed the query, pull a wide net by similarity, then
rerank down to the final top-k. Emits the timing/score detail the query logger
and UI display."""

from __future__ import annotations

from app.config import Settings
from app.interfaces import Embedder, Reranker, VectorStore
from app.logging_config import timed
from app.models import RetrievedChunk


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._reranker = reranker

    def retrieve(
        self, query: str, meeting_id: str | None, timings: dict[str, float]
    ) -> list[RetrievedChunk]:
        with timed(timings, "embed_ms"):
            qvec = self._embedder.embed_one(query)
        with timed(timings, "search_ms"):
            candidates = self._store.search(
                qvec, top_n=self._settings.retrieve_top_n, meeting_id=meeting_id
            )
        with timed(timings, "rerank_ms"):
            return self._reranker.rerank(query, candidates, top_k=self._settings.final_top_k)
