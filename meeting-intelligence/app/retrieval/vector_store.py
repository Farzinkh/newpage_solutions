"""Vector store implementations.

`ChromaVectorStore` is the default: in-process, persistent, zero infra to stand
up. The production upgrade path (named in the README) is pgvector or Qdrant when
corpus size and concurrency grow; that is a new implementation of this same
interface.

`InMemoryVectorStore` is a dependency-free cosine-similarity store used by tests
and the fake profile. Both are idempotent by chunk id.
"""

from __future__ import annotations

import math

from app.interfaces import VectorStore
from app.models import Chunk, RetrievedChunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors, strict=False):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vec

    def search(
        self, query_vector: list[float], top_n: int, meeting_id: str | None = None
    ) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=self._chunks[cid], similarity=_cosine(query_vector, vec))
            for cid, vec in self._vectors.items()
            if meeting_id is None or self._chunks[cid].meeting_id == meeting_id
        ]
        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:top_n]

    def list_meetings(self) -> list[str]:
        return sorted({c.meeting_id for c in self._chunks.values()})

    def count(self) -> int:
        return len(self._chunks)


class ChromaVectorStore(VectorStore):
    def __init__(self, path: str, collection_name: str) -> None:
        import chromadb  # lazy

        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )

    def search(
        self, query_vector: list[float], top_n: int, meeting_id: str | None = None
    ) -> list[RetrievedChunk]:
        where = {"meeting_id": meeting_id} if meeting_id else None
        res = self._col.query(
            query_embeddings=[query_vector], n_results=top_n, where=where
        )
        out: list[RetrievedChunk] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            out.append(
                RetrievedChunk(
                    chunk=Chunk(
                        id=cid,
                        meeting_id=str(meta["meeting_id"]),
                        speaker=str(meta["speaker"]),
                        timestamp=str(meta["timestamp"]),
                        text=doc,
                        turn_index=int(meta["turn_index"]),
                        occurred_at=(
                            str(meta["occurred_at"]) if meta.get("occurred_at") else None
                        ),
                    ),
                    similarity=1.0 - float(dist),  # cosine distance -> similarity
                )
            )
        return out

    def list_meetings(self) -> list[str]:
        got = self._col.get(include=["metadatas"])
        return sorted({str(m["meeting_id"]) for m in got.get("metadatas", [])})

    def count(self) -> int:
        return self._col.count()
