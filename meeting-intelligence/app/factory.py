"""Composition root.

`factory.py` is the *only* place that knows which concrete implementation backs
each seam. Everything else depends on the interfaces. Change a backend in config
and the wiring here picks it up; no other module changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.generation.aggregate import answer_from_items, detect_intent
from app.generation.answerer import Answerer
from app.generation.conversation import recent, rewrite_query
from app.generation.llm import AnthropicLLM, EchoLLM, LLMClient, OpenAILLM
from app.ingestion.item_store import InMemoryItemStore
from app.ingestion.pipeline import IngestionPipeline
from app.interfaces import Embedder, Reranker, VectorStore
from app.logging_config import log_event, timed
from app.models import Answer, HistoryTurn
from app.retrieval.embedder import FakeEmbedder, LocalEmbedder
from app.retrieval.planner import plan_blocks
from app.retrieval.reranker import (
    CohereReranker,
    LocalCrossEncoderReranker,
    NoopReranker,
)
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore, InMemoryVectorStore
from app.transcription.transcriber import FileTranscriber, PlainTextTranscriber

logger = logging.getLogger(__name__)


def build_embedder(s: Settings) -> Embedder:
    if s.embedder_backend == "fake":
        return FakeEmbedder(dim=s.fake_embedding_dim)
    return LocalEmbedder(s.embedding_model)


def build_vector_store(s: Settings) -> VectorStore:
    if s.vector_store_backend == "memory":
        return InMemoryVectorStore()
    return ChromaVectorStore(s.chroma_path, s.collection_name)


def build_reranker(s: Settings) -> Reranker:
    if s.reranker_backend == "local":
        return LocalCrossEncoderReranker(s.cross_encoder_model)
    if s.reranker_backend == "cohere":
        if not s.cohere_api_key:
            raise ValueError("reranker_backend=cohere requires COHERE_API_KEY")
        return CohereReranker(s.cohere_api_key, s.cohere_rerank_model)
    return NoopReranker()


def build_llm(s: Settings) -> LLMClient:
    if s.llm_backend == "openai":
        if not s.openai_api_key:
            raise ValueError("llm_backend=openai requires OPENAI_API_KEY")
        return OpenAILLM(s.openai_api_key, s.openai_model)
    if s.llm_backend == "anthropic":
        if not s.anthropic_api_key:
            raise ValueError("llm_backend=anthropic requires ANTHROPIC_API_KEY")
        return AnthropicLLM(s.anthropic_api_key, s.anthropic_model)
    return EchoLLM()


@dataclass
class Services:
    settings: Settings
    llm: LLMClient
    file_transcriber: FileTranscriber
    voice_transcriber: PlainTextTranscriber
    vector_store: VectorStore
    item_store: InMemoryItemStore
    ingestion: IngestionPipeline
    retriever: Retriever
    answerer: Answerer

    def query(
        self,
        question: str,
        meeting_id: str | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> Answer:
        hist = recent(history or [], self.settings.max_history_turns)
        timings: dict[str, float] = {}
        with timed(timings, "total_ms"):
            # Enumerations (list action items / decisions, summarise) are answered
            # from extracted records, not similarity search — see aggregate.py.
            intent = detect_intent(question)
            if intent is not None:
                answer = answer_from_items(intent, self.item_store.list(meeting_id))
                route = f"aggregate:{intent}"
            else:
                # Two-stage retrieval. Stage 1: resolve any follow-up, pull the
                # wide net, and pick which meeting(s) matter — each contributes
                # its brief. Stage 2: dig the hits + their neighbouring turns.
                search_q = rewrite_query(self.llm, hist, question)
                candidates = self.retriever.candidates(search_q, meeting_id, timings)
                blocks = plan_blocks(
                    self.vector_store, self.item_store, candidates,
                    self.settings, meeting_id,
                )
                focus = (
                    self.answerer.early_focus(question, blocks)
                    if self.settings.two_pass_reasoning else None
                )
                answer = self.answerer.answer_grouped(
                    question, blocks, timings, hist, focus,
                )
                route = f"retrieve:{len(blocks)}mtg"

        chunks = answer.retrieved
        log_event(
            logger,
            "query",
            question=question,
            meeting_id=meeting_id,
            route=route,
            history_turns=len(hist),
            n_retrieved=len(chunks),
            grounded=answer.grounded,
            chunk_ids=[rc.chunk.id for rc in chunks],
            similarities=[round(rc.similarity, 4) for rc in chunks],
            rerank_scores=[rc.rerank_score for rc in chunks],
            timings_ms=timings,
        )
        return answer


def build_services(settings: Settings) -> Services:
    embedder = build_embedder(settings)
    store = build_vector_store(settings)
    item_store = InMemoryItemStore()
    reranker = build_reranker(settings)
    llm = build_llm(settings)
    return Services(
        settings=settings,
        llm=llm,
        file_transcriber=FileTranscriber(),
        voice_transcriber=PlainTextTranscriber(),
        vector_store=store,
        item_store=item_store,
        ingestion=IngestionPipeline(settings, embedder, store, item_store),
        retriever=Retriever(settings, embedder, store, reranker),
        answerer=Answerer(llm),
    )
