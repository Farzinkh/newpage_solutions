"""Reranker implementations.

The embedder is a bi-encoder: it encodes query and chunk separately, so it never
compares them directly and misses fine-grained relevance. A cross-encoder scores
each (query, chunk) pair jointly, so the pattern is: retrieve a wide net (top-N)
cheaply, then rerank and keep the true top-k.

`LocalCrossEncoderReranker` is the default — it runs a small cross-encoder
locally (no API key, cached after first download), so reranking is on by default
at zero cost. `CohereReranker` is the hosted upgrade (better model, external
dependency + per-query latency). `NoopReranker` keeps raw vector order and is the
zero-dependency fallback used by the fake/CI profile.
"""

from __future__ import annotations

from app.interfaces import Reranker
from app.models import RetrievedChunk


class NoopReranker(Reranker):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return candidates[:top_k]


class LocalCrossEncoderReranker(Reranker):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # lazy

        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in candidates])
        ranked = sorted(
            (
                c.model_copy(update={"rerank_score": float(s)})
                for c, s in zip(candidates, scores, strict=False)
            ),
            key=lambda rc: rc.rerank_score,
            reverse=True,
        )
        return ranked[:top_k]


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
