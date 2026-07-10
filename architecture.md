# Architecture

This document is the fuller picture behind the summary in the top-level README.
It covers the two runtime phases, how the two input sources converge, the layer
boundaries that keep the system swappable, the domain model, and the lifecycle
of a single query.

Diagrams are [Mermaid](https://mermaid.js.org/) and render on GitHub.

---

## 1. The two phases

The system has one write path and one read path. Ingestion runs **once per
transcript** (offline-ish); the query path runs **on every question**. They meet
only at the vector store.

```mermaid
flowchart LR
    subgraph ingest["Ingestion — once per transcript"]
        direction LR
        T[Transcript] --> P[Parse turns] --> C[Clean + redact PII] --> K[Chunk per turn] --> E[Embed]
    end

    E --> VS[(Chroma vector store)]

    subgraph query["Query — every question"]
        direction LR
        Q[User question] --> EQ[Embed query] --> R[Retrieve top-N] --> RR["Cohere rerank → top-k"] --> G[Generate answer]
    end

    VS --> R
    G --> A[Answer + citations]
```

The design rule: nothing sensitive crosses into the store. `Clean + redact` runs
*before* `Embed`, so raw PII never reaches Chroma or the LLM.

---

## 2. Two inputs, one pipeline

Uploaded files and browser voice are different at the edge but identical
everywhere after the transcriber seam. A file already carries speaker +
timestamp structure; voice does not, so it is normalised into the same
`Turn` shape (single speaker, synthesised timestamps) before anything shared
runs.

```mermaid
flowchart TD
    F[Transcript file .txt] --> FT[FileTranscriber]
    V[Voice recording<br/>browser Web Speech API] --> VT[PlainTextTranscriber]
    FT --> N[Turns<br/>speaker + timestamp]
    VT --> N
    N --> SP[Shared ingestion pipeline<br/>clean → redact → chunk → embed]
    SP --> VS[(Chroma)]
```

**Diarisation caveat.** Browser STT cannot tell speakers apart, so voice input is
attributed to one speaker. True "who spoke when" needs the raw waveform and a
speaker-embedding model (VAD → x-vectors → clustering, e.g. pyannote), which is
impractical client-side. The intended fix is a server-side transcription backend
(Deepgram / AssemblyAI / Whisper+pyannote) that returns diarised turns and drops
in at the `Transcriber` seam — see §4.

---

## 3. Layers and dependencies

Every layer depends only on the interfaces in `app/interfaces.py`, never on a
concrete vendor. `app/factory.py` is the single composition root that reads
config and picks implementations.

```mermaid
flowchart TD
    API[app/api.py<br/>FastAPI transport] --> FAC[app/factory.py<br/>composition root]
    UI[ui/streamlit_app.py] -. HTTP .-> API

    FAC --> ING[ingestion.pipeline]
    FAC --> RET[retrieval.retriever]
    FAC --> ANS[generation.answerer]

    subgraph seams["Interfaces — the swappable seams"]
        IT[Transcriber]
        IE[Embedder]
        IV[VectorStore]
        IR[Reranker]
        IL[LLMClient]
    end

    ING --> IE & IV
    RET --> IE & IV & IR
    ANS --> IL

    IE --- IEc[LocalEmbedder / FakeEmbedder]
    IV --- IVc[ChromaVectorStore / InMemoryVectorStore]
    IR --- IRc[CohereReranker / NoopReranker]
    IL --- ILc[OpenAILLM / AnthropicLLM / EchoLLM]
    IT --- ITc[FileTranscriber / PlainTextTranscriber]
```

Reading this top to bottom: the transport and orchestration layers know nothing
about MiniLM, Chroma, or Cohere — they hold interface references. The concrete
classes on the bottom row are interchangeable, and which one is live is a config
flag resolved in the factory. The UI is a pure HTTP client and never imports the
core, so it could be replaced wholesale without touching the pipeline.

---

## 4. Swapping an implementation

Because the seams are interfaces, every realistic upgrade is "write a new class,
flip a flag." No pipeline code changes.

| Seam | Default (this build) | Drop-in upgrade | Config flip |
|---|---|---|---|
| `Embedder` | local multilingual MiniLM | hosted embeddings | `EMBEDDER_BACKEND` |
| `VectorStore` | Chroma (in-process) | pgvector / Qdrant | `VECTOR_STORE_BACKEND` |
| `Reranker` | none (`noop`) | Cohere cross-encoder | `RERANKER_BACKEND` |
| `LLMClient` | `echo` (extractive) | OpenAI / Anthropic | `LLM_BACKEND` |
| `Transcriber` | browser Web Speech API | Deepgram / Whisper+pyannote (diarised) | (new backend) |

The `fake` / `memory` / `echo` implementations exist specifically so the whole
system — and its tests — run with zero API keys and no model downloads. That is
what CI uses.

---

## 5. Domain model

The pipeline speaks in these types regardless of which backends are wired in.
Keeping them vendor-free is what lets the seams stay swappable.

```mermaid
classDiagram
    class Turn {
        int index
        str speaker
        str timestamp
        str text
    }
    class Chunk {
        str id
        str meeting_id
        str speaker
        str timestamp
        str text
        int turn_index
    }
    class RetrievedChunk {
        Chunk chunk
        float similarity
        float rerank_score
    }
    class Citation {
        str speaker
        str timestamp
        str chunk_id
        str quote
    }
    class Answer {
        str text
        bool grounded
        List~Citation~ citations
        List~RetrievedChunk~ retrieved
    }
    Turn --> Chunk : chunked into
    Chunk --> RetrievedChunk : scored
    RetrievedChunk --> Answer : cited by
    Citation --> Answer
```

`Chunk.id` is a content hash of `(meeting_id, turn_index, part, text)`. That is
what makes ingestion idempotent: re-ingesting the same transcript produces the
same ids, so the store upserts instead of duplicating.

---

## 6. Query lifecycle

What happens on a single `POST /query`, and where the guardrails and
observability sit.

```mermaid
sequenceDiagram
    participant UI as UI / client
    participant API as FastAPI
    participant R as Retriever
    participant VS as VectorStore
    participant RR as Reranker
    participant AN as Answerer
    participant L as LLMClient

    UI->>API: POST /query {question, meeting_id?}
    API->>R: retrieve(question)
    R->>R: embed query
    R->>VS: search(top_n=20)
    VS-->>R: candidates + similarity
    R->>RR: rerank(candidates, top_k=4)
    RR-->>R: top-k (+ rerank scores)
    R-->>AN: retrieved chunks
    AN->>AN: build grounding prompt<br/>[n] (speaker @ ts) text
    AN->>L: complete(system, user)
    L-->>AN: answer with [n] citations
    AN->>AN: verify citations in-range,<br/>map to (speaker, ts, quote),<br/>set grounded flag
    AN-->>API: Answer
    API-->>UI: {text, citations, retrieved, grounded}
    Note over API: one JSON log line:<br/>chunk_ids, similarities,<br/>rerank_scores, per-stage latency
```

Two guardrail layers appear in the Answerer step: the grounding prompt (answer
only from context, cite `[n]`, or say "Not discussed in the transcript"), and the
output check that drops out-of-range citations and flags an answer as ungrounded
if nothing valid was cited. The retrieval scores and stage latencies returned in
`retrieved` are surfaced in the UI and logged, so "why did it answer that?" is
always inspectable.

---

## 7. Where this goes in production

Summarised from the README's productionising section: the API is already
stateless and config-driven, so scaling is more replicas behind an autoscaler;
Chroma swaps to managed pgvector/Qdrant via the `VectorStore` seam; embedding
moves off the request path onto a queue (ingestion is idempotent, so retries are
safe); secrets move to a manager; and the structured logs feed a tracing/metrics
backend with alerts on grounded-rate and latency, gated in CI by the eval
harness's retrieval metrics.
