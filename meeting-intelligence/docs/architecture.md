# Architecture

The fuller picture behind the summary in the
[project README](../README.md). It covers the two runtime phases, how the two
input sources converge, the layer boundaries that keep the system swappable, the
domain model, and the lifecycle of a single query.

Diagrams are [Mermaid](https://mermaid.js.org/) and render on GitHub.

---

## 1. The two phases

The system has one write path and one read path. Ingestion runs **once per
transcript**; the query path runs **on every question**. They meet at the vector
store and the item/brief store.

```mermaid
flowchart LR
    subgraph ingest["Ingestion — once per transcript"]
        direction TB
        T[Transcript / voice] --> P[Parse turns] --> C[Clean + redact PII]
        C --> TS["Stamp absolute time<br/>(meeting_start + offset)"]
        TS --> K[Chunk per turn] --> E[Embed]
        TS --> X[Extract decisions /<br/>action items → brief]
    end

    E --> VS[(Vector store)]
    X --> IS[(Item + brief store)]

    subgraph query["Query — every question"]
        direction TB
        Q[Question + history] --> RT{aggregation?}
        RT -->|list / summarise| AG[Answer from items]
        RT -->|otherwise| S1["Stage 1: pick meeting(s)<br/>+ inject their briefs"]
        S1 --> S2[Stage 2: dig hits +<br/>before/after context]
        S2 --> G[Generate grounded answer]
    end

    VS --> S1
    IS --> AG
    IS --> S1
    AG --> A[Answer + citations]
    G --> A
```

The design rule: nothing sensitive crosses into the store. `Clean + redact` runs
*before* `Embed` and before extraction, so raw PII never reaches the store or the
LLM.

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
    N --> SP[Shared ingestion pipeline<br/>clean → redact → stamp time →<br/>chunk → embed → extract]
    SP --> VS[(Vector store)]
    SP --> IS[(Item + brief store)]
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

    FAC --> ING[ingestion.pipeline<br/>+ extractor, item_store]
    FAC --> RET[retrieval.retriever<br/>+ planner]
    FAC --> ANS[generation.answerer<br/>+ conversation, aggregate]

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
    IR --- IRc[LocalCrossEncoder / Cohere / Noop]
    IL --- ILc[OpenAILLM / AnthropicLLM / EchoLLM]
    IT --- ITc[FileTranscriber / PlainTextTranscriber]
```

Reading this top to bottom: the transport and orchestration layers know nothing
about MiniLM, Chroma, or Cohere — they hold interface references. The concrete
classes on the bottom row are interchangeable, and which one is live is a config
flag resolved in the factory. The UI is a pure HTTP client and never imports the
core, so it could be replaced wholesale without touching the pipeline.

The item/brief store (`ingestion.item_store`) holds the per-meeting decisions,
action items, and brief derived at ingestion. It sits beside the vector store
because those records are used by *enumeration* (aggregation answers) and as
whole-meeting context, not by similarity search.

---

## 4. Swapping an implementation

Because the seams are interfaces, every realistic upgrade is "write a new class,
flip a flag." No pipeline code changes.

| Seam | Default (this build) | Drop-in upgrade | Config flip |
|---|---|---|---|
| `Embedder` | local multilingual MiniLM | hosted embeddings | `EMBEDDER_BACKEND` |
| `VectorStore` | Chroma (in-process) | pgvector / Qdrant | `VECTOR_STORE_BACKEND` |
| `Reranker` | `noop` (measured: the local cross-encoder didn't help this corpus) | local cross-encoder / Cohere | `RERANKER_BACKEND` |
| `LLMClient` | `echo` (extractive) | OpenAI / Anthropic | `LLM_BACKEND` |
| `Transcriber` | browser Web Speech API | Deepgram / Whisper+pyannote (diarised) | (new backend) |

The `fake` / `memory` / `echo` implementations exist specifically so the whole
system — and its tests — run with zero API keys and no model downloads. That is
what CI and the [zero-key demo](../docker-compose.demo.yml) use.

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
        str occurred_at
    }
    class Chunk {
        str id
        str meeting_id
        str speaker
        str timestamp
        str text
        int turn_index
        str occurred_at
        display_time()
    }
    class RetrievedChunk {
        Chunk chunk
        float similarity
        float rerank_score
    }
    class ExtractedItem {
        str meeting_id
        str kind
        str speaker
        str timestamp
        str text
        str occurred_at
    }
    class MeetingBrief {
        str meeting_id
        List~str~ participants
        List~str~ decisions
        List~str~ action_items
    }
    class Citation {
        str meeting_id
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
    Turn --> ExtractedItem : tagged as
    Chunk --> RetrievedChunk : scored
    RetrievedChunk --> Answer : cited by
    ExtractedItem --> MeetingBrief : summarised into
    Citation --> Answer
```

`Chunk.id` is a content hash of `(meeting_id, turn_index, part, text)`. That is
what makes ingestion idempotent: re-ingesting the same transcript produces the
same ids, so the store upserts instead of duplicating.

`occurred_at` is the absolute wall-clock time (`meeting_start + relative
offset`), set when the meeting's start is known (from the file name or an
explicit `started_at`). `display_time()` returns it when present and falls back
to the relative timestamp otherwise, so citations are unambiguous across
meetings. `HistoryTurn` (role, content) carries prior conversation turns for
follow-up resolution.

---

## 6. Query lifecycle

What happens on a single `POST /query`, and where the routing, guardrails, and
observability sit.

```mermaid
sequenceDiagram
    participant UI as UI / client
    participant API as FastAPI
    participant SV as Services.query
    participant R as Retriever
    participant PL as Planner
    participant IS as Item/Brief store
    participant AN as Answerer
    participant L as LLMClient

    UI->>API: POST /query {question, meeting_id?, history?}
    API->>SV: query(...)
    alt aggregation intent (list / summarise / decisions)
        SV->>IS: list items (optionally scoped)
        IS-->>SV: decisions / action items
        SV-->>API: Answer (enumerated, each cited)
    else retrieval
        SV->>L: rewrite follow-up → standalone query
        SV->>R: candidates(query) — embed + search top_n
        R-->>SV: wide-net candidates + similarity
        SV->>PL: plan_blocks (group by meeting, pick top meetings,<br/>expand hits with neighbours)
        PL->>IS: brief per selected meeting
        IS-->>PL: briefs
        PL-->>SV: per-meeting blocks (briefs + hits + context)
        SV->>AN: answer_grouped(blocks, history)
        AN->>L: complete(system, grouped prompt)
        L-->>AN: answer with [n] citations
        AN->>AN: verify [n] in range → map to<br/>(meeting, speaker, abs-time, quote),<br/>set grounded flag
        AN-->>SV: Answer
    end
    SV-->>API: Answer
    API-->>UI: {text, citations, retrieved, grounded}
    Note over SV: one JSON log line: route, history_turns,<br/>chunk_ids, similarities, rerank_scores,<br/>grounded, per-stage latency
```

Two guardrail layers appear in the Answerer step: the grounding prompt (answer
only from the numbered excerpts, cite `[n]`, never cite the brief/history, or say
"Not discussed in the transcript"), and the output check that drops out-of-range
citations and flags an answer as ungrounded if nothing valid was cited. The
retrieval scores and stage latencies returned in `retrieved` are surfaced in the
UI and logged, so "why did it answer that?" is always inspectable.

---

## 7. Where this goes in production

Summarised from the README's productionising section: the API is already
stateless and config-driven, so scaling is more replicas behind an autoscaler;
Chroma swaps to managed pgvector/Qdrant via the `VectorStore` seam (which also
becomes the natural home for the item/brief store); embedding moves off the
request path onto a queue (ingestion is idempotent, so retries are safe); secrets
move to a manager; and the structured logs feed a tracing/metrics backend with
alerts on grounded-rate and latency, gated in CI by the eval harness's retrieval
metrics.
