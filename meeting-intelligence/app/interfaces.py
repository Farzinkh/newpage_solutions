"""The swappable seams.

Every external dependency the system has — turning speech to text, turning text
to vectors, storing/searching those vectors, reranking, and generating the
answer — sits behind one of these interfaces. Concrete implementations live in
their own modules and are chosen at runtime by `factory.py` from config.

This is the single most important design decision in the codebase: swapping the
browser transcriber for Deepgram, MiniLM for OpenAI embeddings, or Chroma for
pgvector is a config change and a new class, never a change to the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import Chunk, RetrievedChunk, Turn


class Transcriber(ABC):
    """Speech/raw input -> normalised speaker turns.

    The browser default produces a single-speaker stream; server-side backends
    (Deepgram, Whisper+pyannote) return real diarised turns. Either way the
    output is `list[Turn]`, so the rest of the pipeline never knows which ran.
    """

    @abstractmethod
    def to_turns(self, raw_text: str) -> list[Turn]: ...


class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Idempotent by chunk id: re-ingesting the same content updates in
        place rather than duplicating."""

    @abstractmethod
    def search(
        self, query_vector: list[float], top_n: int, meeting_id: str | None = None
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def list_meetings(self) -> list[str]: ...

    @abstractmethod
    def count(self) -> int: ...


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class LLMClient(ABC):
    # Whether this client can perform free-form generation (query rewriting,
    # abstractive answers). The extractive EchoLLM used by the keyless demo
    # cannot, so features that need real generation degrade gracefully instead
    # of feeding it prompts it would mangle.
    can_generate: bool = True

    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...
