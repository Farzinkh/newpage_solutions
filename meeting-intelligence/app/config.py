"""Configuration surface.

Everything tunable lives here and is overridable via environment variables or a
`.env` file — no magic constants scattered through the code. The `*_backend`
fields are how you flip a seam between a real implementation and a fake one
(fakes let the whole app and its tests run with zero API keys, which is what CI
uses).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "meeting-intelligence"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Seam selection -----------------------------------------------------
    embedder_backend: Literal["local", "fake"] = "local"
    vector_store_backend: Literal["chroma", "memory"] = "chroma"
    reranker_backend: Literal["cohere", "noop"] = "noop"
    llm_backend: Literal["openai", "anthropic", "echo"] = "echo"

    # --- Embedding ----------------------------------------------------------
    # Default is multilingual so non-English transcripts stay in their own
    # language and citations remain faithful (no lossy translate-to-English).
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fake_embedding_dim: int = 64

    # --- Vector store -------------------------------------------------------
    chroma_path: str = "./.chroma"
    collection_name: str = "meeting_turns"

    # --- Reranker -----------------------------------------------------------
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-english-v3.0"

    # --- LLM ----------------------------------------------------------------
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"

    # --- Retrieval knobs (justified by the eval harness, not by vibes) ------
    retrieve_top_n: int = 20  # wide net for the reranker
    final_top_k: int = 4  # what the LLM actually sees

    # --- Ingestion ----------------------------------------------------------
    max_chunk_chars: int = 1200  # split turns longer than this on sentences
    redaction_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    """Cached so the app builds services once. Tests clear the cache or
    construct `Settings(...)` directly with fake backends."""
    return Settings()
