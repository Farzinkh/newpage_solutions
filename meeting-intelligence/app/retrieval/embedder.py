"""Embedder implementations.

`LocalEmbedder` runs sentence-transformers locally (default: a multilingual
MiniLM) — free, no API key, keeps document/query in their original language.
Heavy imports are lazy so the package imports without the ML stack installed.

`FakeEmbedder` is deterministic (hash -> fixed vector). It has no learned
semantics, but it is reproducible and dependency-free, which is exactly what
unit tests and CI need. Selected via `embedder_backend=fake`.
"""

from __future__ import annotations

import hashlib
import math

from app.interfaces import Embedder


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder for tests.

    Not semantically meaningful, but words that overlap between a query and a
    chunk push their vectors together, which is enough to exercise retrieval
    ordering deterministically.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            v[h % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]
