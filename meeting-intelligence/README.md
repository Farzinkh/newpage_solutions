# Meeting Intelligence

A conversational assistant that answers questions about meeting transcripts —
decisions, action items, "what did we say about X" — with answers grounded in
the transcript and cited back to the speaker and timestamp. Supports both
uploaded transcript files and browser voice-to-text.

This is **Option 3** from the assignment. The README is the part I care most
about: the code is the easy half; the reasoning below is the argument for *why*
it's built this way.

---

## Quick start

The defaults run **key-free** (a fake embedder, in-memory store, and an
extractive "echo" LLM), so you can see the whole pipeline work before wiring up
any accounts.

```bash
# 1. Fake profile — no keys, no model downloads, instant.
EMBEDDER_BACKEND=fake VECTOR_STORE_BACKEND=memory RERANKER_BACKEND=noop LLM_BACKEND=echo \
  uvicorn app.api:app --port 8000

# 2. Tests + retrieval eval (also key-free)
make test
make eval

# 3. Real profile — local embeddings, Chroma, and a hosted LLM
pip install -e ".[dev,openai,ui]"
cp .env.example .env            # set EMBEDDER_BACKEND=local, LLM_BACKEND=openai, OPENAI_API_KEY=...
make api                        # backend on :8000
make ui                         # Streamlit on :8501 (separate terminal)
python scripts/seed.py          # load the three sample meetings
```

Docker: `docker compose up --build` starts the API (`:8000`) and UI (`:8501`).

---

## What it does

- **Ingests** transcripts (`[HH:MM:SS] Speaker: text`) or browser voice input.
- **Redacts** leaked PII (emails, phones, SSNs, card numbers) before storage.
- **Retrieves** relevant turns and **generates** an answer that cites its sources.
- **Refuses to hallucinate**: if the transcript doesn't cover it, it says so.
- **Shows its work**: every answer carries the retrieved chunks and their scores.

---

## Architecture

```
                 ┌── file  ──> FileTranscriber ──┐
   input ────────┤                               ├──> Turns ──> preprocess (clean + redact PII)
                 └── voice ──> PlainTextTranscriber┘                       │
                                                                           ▼
                                                            chunk per speaker turn
                                                                           │
                                                                           ▼
                                                        embed ──> Chroma (vector store)
                                                                           │
   question ──> embed ──> search top-N ──> Cohere rerank ──> top-k ────────┘
                                                     │
                                                     ▼
                              LLM (grounding prompt) ──> answer + citations
```

Ingestion runs once per transcript; the query path runs per question. Both input
sources converge to `Turns` at the transcriber seam, so everything after it is
shared. See [`docs/architecture.md`](docs/architecture.md) for the fuller
diagrams — the dual-input flow, layer dependencies, domain model, and the query
lifecycle.

### Module layout

```
app/
  interfaces.py     # the swappable seams (ABCs): Transcriber, Embedder,
                    #   VectorStore, Reranker, LLMClient
  config.py         # all knobs (pydantic-settings), env-overridable
  factory.py        # the ONLY place that picks concrete implementations
  models.py         # Turn / Chunk / RetrievedChunk / Citation / Answer
  ingestion/        # parser, redactor, chunker, pipeline (idempotent)
  transcription/    # file + voice -> Turns
  retrieval/        # embedder, vector_store, reranker, retriever
  generation/       # llm clients, answerer (prompt + citation guardrail)
  api.py            # FastAPI transport
ui/                 # Streamlit client (talks to the API, never imports core)
eval/               # gold set + retrieval metrics (hit-rate@k, MRR)
tests/              # unit + integration, all runnable with zero keys
```

The design principle throughout: **the pipeline depends on interfaces, never on
vendors.** Swapping MiniLM for OpenAI embeddings, Chroma for pgvector, or the
browser transcriber for Deepgram is a new class plus a config flag — no pipeline
code changes. `factory.py` is the single composition root that knows the mapping.

---

## RAG / LLM approach and decisions

**Chunking — per speaker turn, not fixed windows.** A turn is a natural semantic
unit in a meeting, and it lets every chunk carry `speaker` + `timestamp` as
metadata, which is what makes citations possible. Fixed-token chunking would cut
across turns and lose that structure. Turns longer than `MAX_CHUNK_CHARS` are
split on sentence boundaries so a monologue doesn't become one giant chunk.

**Embedding — local multilingual MiniLM by default.** Free, no API key, and runs
offline after the first download. I chose a *multilingual* model deliberately:
the naive approach to non-English input is "translate to English then embed," but
that's lossy and — worse — breaks citations, because the stored text no longer
matches what was actually said. A multilingual model keeps documents and queries
in their own language in one shared vector space, so citations stay faithful. A
hosted embedding model would retrieve marginally better; I traded that for zero
cost and zero key. Swappable via `EMBEDDER_BACKEND`.

**Vector store — Chroma, in-process + persistent.** Zero infra to stand up, which
fits "start simple." The production upgrade path is pgvector or Qdrant once
corpus size and concurrency grow — that's a new `VectorStore` implementation, not
a rewrite.

**Retrieval — wide net, then rerank.** The embedder is a *bi-encoder*: it encodes
the query and each chunk separately, so it never compares them directly and misses
fine-grained relevance. So the pattern is: retrieve top-N (=20) cheaply by cosine
similarity, then rerank with **Cohere Rerank** (a *cross-encoder* that scores each
query–chunk pair jointly) down to the top-k (=4) the LLM sees. The cost of that
quality is an external API dependency and per-query latency, so reranking is
behind a flag (`RERANKER_BACKEND`) and defaults off. Both similarity and rerank
scores are logged and shown in the UI.

**Orchestration — no framework.** I wrote the ~150 lines of RAG glue directly
rather than pulling in LangChain/LlamaIndex. At this scope a framework's
abstractions cost more legibility than they save, and hand-rolled code is easier
for a reviewer to read and for me to reason about. The interfaces give me the
composability a framework would, without the indirection. (For a much larger
system with many loaders and tools, that calculus flips.)

**Prompt & context management.** The LLM gets a numbered list of retrieved
excerpts, each prefixed with `[n] (speaker @ timestamp)`, and a system prompt
that says: answer only from these excerpts, cite as `[n]`, and if the answer
isn't there, say exactly "Not discussed in the transcript." `temperature=0` for
determinism.

**Guardrails — two layers.** (1) The grounding prompt above. (2) An *output*
check: I parse the `[n]` citations the model emits, drop any that point outside
the retrieved set, and map the valid ones back to real (speaker, timestamp,
quote) records. If the model answered but cited nothing valid, the answer is
flagged `grounded: false` rather than presented as sourced. This catches the
failure mode where a model writes a confident answer with a fabricated citation.

**PII redaction.** Runs at ingestion, before embedding, so nothing sensitive ever
reaches the store or the LLM. The key judgment call: I redact *leaked* PII
(emails, phones, SSNs, cards) but **keep speaker names**, because meetings are
about people — blanket-redacting names would break "what did Sarah commit to?".
Redaction is consistent across the stored chunk, its embedding, and the displayed
citation. Placeholders are typed (`[EMAIL]`) not blanks, so the sentence still
reads naturally for the embedder. The regex layer is a baseline; Presidio (spaCy
NER) is the documented upgrade for names-in-context and loads lazily if installed.

**Observability.** Every query logs one JSON line: the question, retrieved chunk
ids, similarity and rerank scores, grounded flag, and per-stage latency
(embed / search / rerank / llm / total). The UI surfaces the same scores under
each answer. Only redaction *counts* are logged, never raw PII values.

**Quality — a real eval harness.** `eval/gold_set.json` maps hand-curated
questions to the turns that answer them; `eval/evaluate.py` computes hit-rate@k
and MRR. This is what makes every knob (k, rerank on/off, embedding model) a
*measured* decision instead of a vibe. Run it before and after a change and
compare. (With the fake embedder the numbers are only a plumbing check — point it
at the local or hosted embedder for meaningful scores.)

---

## Voice-to-transcript

The bonus. The browser records and transcribes locally via the Web Speech API
(`streamlit-mic-recorder`); only the recognised text is sent to the backend,
where it runs through the exact same pipeline as a file. The one honest caveat:
browser STT has **no diarisation** — it can't tell speakers apart — so voice
input is attributed to a single speaker with synthesised timestamps.

Real diarisation ("who spoke when") needs the raw waveform and a speaker-embedding
model (VAD → x-vectors → clustering, e.g. pyannote), which is impractical
client-side. Rather than build a browser DSP stage, I treat diarisation as a
property of the transcription backend: the `Transcriber` seam lets a server-side
provider (Deepgram, AssemblyAI, Whisper+pyannote) that returns diarised turns
drop in without touching anything downstream. So the upgrade isn't "nicer text,"
it's "voice finally produces the same first-class speaker turns a file gives you."

---

## Productionising this

What I'd need to deploy this on a hyperscaler and make it scale:

- **Stateless API behind autoscaling** (ECS/Cloud Run/GKE) — the app already
  builds services from config, so horizontal scaling is just more replicas.
- **Managed vector store**: swap Chroma → pgvector (RDS/Cloud SQL) or a managed
  Qdrant/Pinecone. The `VectorStore` interface already isolates this.
- **Async ingestion**: move embedding off the request path onto a queue
  (SQS + workers) so large transcripts don't block; ingestion is idempotent, so
  retries are safe.
- **Secrets** in a manager (Secrets Manager / KMS), never env files in prod.
- **AuthN/Z + tenancy**: meetings scoped per user/org; today `meeting_id`
  filtering is the only isolation.
- **Observability**: ship the structured logs to CloudWatch/Datadog, add tracing
  (OpenTelemetry) across the retrieve/rerank/generate stages, and alert on
  grounded-rate and latency.
- **Caching**: cache embeddings for repeated ingests and (optionally) answers.
- **CI/CD**: run `ruff` + `pytest` + the eval harness on every PR, with a
  regression gate on hit-rate/MRR.

---

## Engineering standards followed (and skipped)

**Followed:** typed throughout (pydantic + type hints); clear layering with
dependency-inverted seams; config over constants; idempotent ingestion; unit
tests on the deterministic logic (parser, redactor, chunker) plus end-to-end API
tests, all runnable with zero keys; structured logging; a Dockerfile (non-root,
healthcheck) and compose; ruff config; a Makefile for the common flows.

**Skipped (deliberately, with reasons):** no auth/multi-tenancy (out of scope for
the exercise); PII detection is regex-based, so it misses names-in-context —
Presidio is wired as an optional upgrade but not the default; the eval set is
small and hand-made, not a statistically meaningful benchmark; no retry/backoff
around the hosted APIs yet; the UI is Streamlit — I chose speed over polish since
the brief said UI polish isn't the bar, and the API is the real product surface.

---

## How I used AI tools

I used an AI assistant as a **design partner and a fast typist**, not an author.
The workflow: I drove the architecture decisions in conversation — the seam-based
design, per-turn chunking, the redaction judgment call (keep names, redact
contact PII), rerank-as-cross-encoder, and the diarisation-belongs-in-the-backend
call are mine, and several were places where I *rejected* a plausible-sounding
suggestion (translate-to-English embedding, stopword removal, browser-side FFT
diarisation) because they were subtly wrong. The AI was useful for scaffolding
boilerplate (pydantic models, the Chroma adapter, test skeletons) and for
pressure-testing my reasoning.

My do's and don'ts, learned here: **do** use it to draft the mechanical parts and
to argue against your own design; **don't** let it make the judgment calls or
write the README's reasoning — that has to be yours, or the whole point (showing
how *you* think) is lost. Everything it produced I read, edited, and can defend
line by line. This README is written by me about my own decisions.

---

## What I'd do differently / add next

In rough priority order:
1. **Turn on reranking by default** and re-run the eval to quantify the lift —
   right now it's wired but off.
2. **Add hybrid retrieval** (dense + BM25). Meetings are full of exact terms
   (names, dates, project codenames, dollar figures) that keyword search nails
   and pure semantic search sometimes fuzzes. For this corpus I'd expect this to
   help more than reranking.
3. **Action-item / decision extraction** as a structured pass at ingestion, so
   "list the action items" is answered from extracted records rather than
   similarity search — those queries are extraction, not retrieval.
4. **Presidio** as the default redactor for names-in-context.
5. **Streaming responses** and answer caching.
6. **Server-side diarised transcription** backend behind the `Transcriber` seam.
7. **Grow the eval set** and gate CI on retrieval-metric regressions.
