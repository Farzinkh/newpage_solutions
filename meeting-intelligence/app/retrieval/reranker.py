"""Reranker implementations.

The embedder is a bi-encoder: it encodes query and chunk separately, so it never
compares them directly and misses fine-grained relevance. `CohereReranker` is a
cross-encoder — it scores each (query, chunk) pair jointly — so the pattern is
retrieve a wide net (top-N) cheaply, then rerank and keep the true top-k.

Cost of that quality: an external API dependency and per-query latency. So it is
behind a flag (`reranker_backend`). `NoopReranker` keeps vector order and is the
zero-dependency default.
"""

from __future__ import annotations

from app.interfaces import Reranker
from app.models import RetrievedChunk


class NoopReranker(Reranker):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return candidates[:top_k]


class CohereReranker(Reranker):
    def __init__(self, api_key: str, model: str) -> None:
        import cohere  # lazy

        self._client = cohere.Client(api_key)
        self._model = model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        res = self._client.rerank(
            model=self._model,
            query=query,
            documents=[c.chunk.text for c in candidates],
            top_n=min(top_k, len(candidates)),
        )
        out: list[RetrievedChunk] = []
        for item in res.results:
            rc = candidates[item.index]
            out.append(rc.model_copy(update={"rerank_score": item.relevance_score}))
        return out
