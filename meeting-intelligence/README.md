# Meeting Intelligence

A conversational assistant that answers questions about meeting transcripts:
decisions, action items, "what did we say about X", and so on. Answers are
grounded in the transcript and cited back to the speaker, the meeting, and the
(absolute) timestamp. It handles questions that reach across several meetings,
and it takes both uploaded transcript files and browser voice-to-text.

This is **Option 3** from the assignment. The README is the part I care most
about. The code is the easy half; what follows is the argument for *why* it's
built the way it is.

---

## Quick start

The defaults run **key-free** (a fake embedder, an in-memory store, and an
extractive "echo" LLM), so you can watch the whole pipeline work before wiring up
any accounts.

```bash
# 1. Fake profile: no keys, no model downloads, instant.
EMBEDDER_BACKEND=fake VECTOR_STORE_BACKEND=memory RERANKER_BACKEND=noop LLM_BACKEND=echo \
  uvicorn app.api:app --port 8000

# 2. Tests + retrieval eval (also key-free)
make test
make eval

# 3. Real profile: local embeddings, Chroma, and a hosted LLM
pip install -e ".[dev,openai,ui]"
cp .env.example .env            # set EMBEDDER_BACKEND=local, LLM_BACKEND=openai, OPENAI_API_KEY=...
make api                        # backend on :8000
make ui                         # Streamlit on :8501 (separate terminal)
python scripts/seed.py          # load the three sample meetings
```

Docker:
- **Zero-key demo** (fake backends, auto-seeded, lightweight image):
  `docker compose -f docker-compose.demo.yml up --build`. The UI comes up on
  **http://localhost:8501** and the API on `:8000`. Click an example prompt to
  see a cited answer.
- **Real profile** (local embeddings + Chroma + a hosted LLM):
  `docker compose up --build`. UI on `:8501`, API on `:8000`.

---

## What it does

- **Ingests** transcripts (`[HH:MM:SS] Speaker: text`, or plain `Speaker: text`)
  or browser voice input.
- **Redacts** leaked PII (emails, phones, SSNs, card numbers) before anything is
  stored.
- **Extracts** decisions and action items at ingestion, so "list the action
  items" or "summarise the meeting" is answered by enumeration rather than top-k
  search.
- **Retrieves** the relevant turns and **generates** an answer that cites its
  sources. It also injects the **whole-meeting brief** (participants, decisions,
  action items) as context, so answers aren't stuck with isolated fragments.
- **Is conversational**: prior turns are fed back, and a follow-up like "who owns
  that?" is rewritten into a standalone query before retrieval.
- **Won't hallucinate**: if the transcript doesn't cover something, it says so.
- **Shows its work**: every answer carries the retrieved chunks and their scores.

---

## Architecture

```
   input ──(file | voice)──> Turns ──> preprocess ──> ┌─ chunk ──> embed ──> vector store
                                    (clean + redact +  │
                                     stamp abs. time)  └─ extract decisions /
                                                          action items ──> item + brief store

   question + history ──> aggregation? ──yes──> answer from items (enumerated, cited)
                              │ no
                              ▼
        stage 1: pick meeting(s) + inject their briefs
                              │
                              ▼
        stage 2: dig top hits + their before/after context
                              │
                              ▼
        LLM (grounded, grouped-by-meeting prompt) ──> answer + citations
```

Ingestion runs once per transcript; the query path runs per question. Both input
sources meet at `Turns` at the transcriber seam, so everything after that point
is shared. Retrieval works in two stages: pick the meetings that matter, then dig
into each one with its neighbouring turns. Reranking is wired up but off by
default, because the eval showed it didn't help (more on that below). The fuller
Mermaid diagrams live in [`docs/architecture.md`](docs/architecture.md): the
dual-input flow, the layer dependencies, the domain model, and the two-stage
query lifecycle.

### Module layout

```
app/
  interfaces.py     # the swappable seams (ABCs): Transcriber, Embedder,
                    #   VectorStore, Reranker, LLMClient
  config.py         # all knobs (pydantic-settings), env-overridable
  factory.py        # the ONLY place that picks concrete implementations
  models.py         # Turn / Chunk / RetrievedChunk / Citation / Answer
                    #   ExtractedItem / MeetingBrief / HistoryTurn
  ingestion/        # parser, redactor, chunker, extractor, timestamps,
                    #   item_store, pipeline
  transcription/    # file + voice -> Turns
  retrieval/        # embedder, vector_store, reranker, retriever,
                    #   planner (two-stage cross-meeting + neighbour expansion)
  generation/       # llm clients, answerer, conversation (rewrite), aggregate
  api.py            # FastAPI transport (ingest/query/items/brief/meetings)
ui/                 # Streamlit client (talks to the API, never imports core)
eval/               # gold set + retrieval metrics (hit-rate@k, MRR)
tests/              # unit + integration, all runnable with zero keys
```

The principle running through all of it: **the pipeline depends on interfaces,
never on vendors.** Swapping MiniLM for OpenAI embeddings, Chroma for pgvector, or
the browser transcriber for Deepgram is a new class and a config flag, with no
changes to pipeline code. `factory.py` is the one composition root that knows the
mapping.

---

## RAG / LLM approach and decisions

**Chunking: per speaker turn, not fixed windows.** A turn is a natural semantic
unit in a meeting, and it lets every chunk carry `speaker` and `timestamp` as
metadata, which is what makes citations possible. Fixed-token chunking would cut
across turns and throw that structure away. Turns longer than `MAX_CHUNK_CHARS`
get split on sentence boundaries so a monologue doesn't become one giant chunk.

**Embedding: local multilingual MiniLM by default.** It's free, needs no API key,
and runs offline after the first download. I picked a *multilingual* model on
purpose. The naive way to handle non-English input is "translate to English then
embed", but that's lossy, and worse, it breaks citations, because the stored text
no longer matches what was actually said. A multilingual model keeps documents and
queries in their own language in one shared vector space, so citations stay
faithful. A hosted embedding model would retrieve slightly better; I traded that
for zero cost and zero keys. It's swappable via `EMBEDDER_BACKEND`.

**Vector store: Chroma, in-process and persistent.** There's no infra to stand
up, which fits "start simple". The production upgrade is pgvector or Qdrant once
the corpus and concurrency grow, and that's a new `VectorStore` implementation
rather than a rewrite.

**Retrieval: wide net, then optional rerank, and a measurement that changed my
mind.** The embedder is a *bi-encoder*: it encodes the query and each chunk
separately, so it never compares them directly and can miss fine-grained
relevance. The textbook fix is to retrieve the top-N (=20) cheaply by cosine
similarity, then rerank with a *cross-encoder* (which scores each query/chunk pair
jointly) down to the top-k (=4) that the LLM sees. I wired up **two** rerankers, a
keyless local cross-encoder (`ms-marco-MiniLM`) and hosted **Cohere Rerank**, and
I fully expected to turn reranking on by default.

Then I ran the eval, and it talked me out of it. On this corpus the local
cross-encoder made retrieval **worse**: hit-rate@4 fell from 1.0 to 0.875 and MRR
from 0.635 to 0.521, because it pushed the correct chunk below rank 4 for one
query. The small MS-MARCO model is trained on web-passage ranking, and short
conversational meeting turns are out of its distribution, while the multilingual
bi-encoder already gets top-4 recall right here. So the honest call was to keep
reranking **wired, tested, and behind a flag** (`RERANKER_BACKEND`) but **off by
default**, because the numbers say it doesn't help *this* data. That's the whole
point of having an eval harness: it turns a plausible default into a measured one.
I'd revisit it with a larger, domain-tuned reranker and a bigger eval set. Both
scores are logged and shown in the UI.

**Orchestration: no framework.** I wrote the ~150 lines of RAG glue by hand rather
than pulling in LangChain or LlamaIndex. At this size a framework's abstractions
cost more readability than they buy, and hand-written code is easier for a
reviewer to read and for me to reason about. The interfaces give me the
composability a framework would, without the indirection. (For a much bigger
system with many loaders and tools, that trade-off flips.)

**Beyond top-k: the parts that make it a *meeting* assistant rather than a generic
RAG box.** Plain chunk retrieval has a few failure modes that hurt meetings in
particular, and each one gets its own mechanism:

1. **Aggregation queries are extraction, not retrieval.** "What are all the action
   items?" needs *every* relevant turn, but top-k only returns a handful (the eval
   shows a 3-turn action-item answer retrieving 1 of 3). So a deterministic,
   keyless pass at ingestion tags decisions and action items using high-precision
   cue phrases. I tuned those toward precision and dropped broad cues like a bare
   "let's", which had been tagging "let's wrap" as a to-do. An intent detector
   routes enumerations and summaries to answer straight from those records, so the
   list is *complete* and every line is cited. The natural upgrade is an LLM
   extraction pass behind the same `ExtractedItem` seam.

2. **Isolated chunks lose the meeting's narrative, so retrieval is two-stage.** A
   turn on its own doesn't know the meeting decided single-device-only, or that
   Daniel owns the doc, and a flat top-k across meetings can't tell which meeting
   to trust. So the query path is coarse-to-fine (`retrieval/planner.py`):
   - *Stage 1 (which meetings):* group the wide-net candidates by meeting, rank
     meetings by their best hit, and keep the top few (`MAX_MEETINGS`). Each
     selected meeting contributes its **brief** (participants, decisions, action
     items, all derived once at ingestion) as clearly marked, *non-citable*
     orientation.
   - *Stage 2 (dig in):* for each meeting take its top hits (`PER_MEETING_HITS`)
     and expand each one with `CONTEXT_WINDOW` turns **before and after** it. That
     expansion is a metadata lookup by `turn_index`, not a second similarity
     search, so the model reads each quote *in the flow of the conversation*.

   The answer prompt is then grouped by meeting. Each block is an overview
   followed by its excerpts (the hits and their `(context)` neighbours), with
   every turn numbered and citable. So a question that spans several meetings gets
   both the whole-meeting shape of *each* relevant meeting and the local context
   around each hit, instead of four orphaned fragments. An optional first LLM pass
   (`TWO_PASS_REASONING`) reads the briefs and forms an early hypothesis about
   which meeting matters before the answer pass. (`/brief` exposes a brief
   directly.)

3. **A one-shot Q&A box isn't "conversational".** The brief asks for a
   *conversational* system, so `/query` accepts prior turns. A follow-up like "who
   owns that?" is first **rewritten into a standalone question** (with a real LLM;
   for the keyless EchoLLM it falls back to carrying the previous question's terms)
   so retrieval has the referent, and the recent turns are also passed to the
   answer prompt so it can resolve references. It's capped at `MAX_HISTORY_TURNS`.

4. **Cross-meeting questions need attribution and absolute time.** A question can
   pull turns from several meetings at once, since search is unscoped by default.
   Two things keep that coherent instead of confusing. First, every excerpt,
   citation, and extracted item is **labelled with its meeting**. Second, relative
   transcript timestamps (`00:02:36`, which collide across meetings) are anchored
   to **wall-clock datetimes** at ingestion, as `meeting_start + offset`, where
   `meeting_start` comes from the file name (`name_2026-06-03_1548.txt`) or an
   explicit `started_at`. That makes `[2] (incident_retro · Omar @ 2026-06-05
   10:15:33)` globally unambiguous. It's a strictly additive step: with no date in
   the name it just falls back to the relative timestamp. (The parser also strips
   the date and time back off to recover a stable `meeting_id`, so the naming
   convention never leaks into identifiers or the eval.)

**Prompt and context management.** The LLM gets, in order: the recent conversation
if there is any, then the retrieved context **grouped by meeting**. Each meeting
block is its non-citable brief followed by numbered excerpts, each prefixed with
`[n] (meeting · speaker @ time)` and marked `(context)` when it's a before/after
neighbour rather than a hit. The system prompt tells it to answer only from the
numbered excerpts, cite as `[n]`, never cite the brief or the history, and if the
answer isn't there to reply exactly "Not discussed in the transcript".
`temperature=0` keeps it deterministic.

**Guardrails, in two layers.** First, the grounding prompt above. Second, an
*output* check: I parse the `[n]` citations the model emits, drop any that point
outside the retrieved set, and map the valid ones back to real (speaker,
timestamp, quote) records. If the model answered but cited nothing valid, the
answer is flagged `grounded: false` rather than presented as sourced. That catches
the case where a model writes a confident answer with a made-up citation.

**PII redaction.** This runs at ingestion, before embedding, so nothing sensitive
ever reaches the store or the LLM. The judgment call was what to redact: I redact
*leaked* PII (emails, phones, SSNs, cards) but **keep speaker names**, because
meetings are about people and blanket-redacting names would break "what did Sarah
commit to?". The redaction is consistent across the stored chunk, its embedding,
and the displayed citation. Placeholders are typed (`[EMAIL]`) rather than blanks,
so the sentence still reads naturally for the embedder. The regex layer is a
baseline; Presidio (spaCy NER) is the documented upgrade for names-in-context, and
it loads lazily if installed.

**Observability.** Every query logs one JSON line: the question, the route it took
(aggregation vs. retrieval, and how many meetings it spanned), the retrieved chunk
ids, the similarity and rerank scores, the grounded flag, and per-stage latency
(embed, search, llm, total). The UI surfaces the same scores under each answer.
Only redaction *counts* are logged, never the raw PII values.

**Quality: a real eval harness.** `eval/gold_set.json` maps hand-curated questions
to the turns that answer them, and `eval/evaluate.py` computes hit-rate@k and MRR.
That's what turns every knob (k, rerank on or off, embedding model) into a
*measured* decision instead of a guess. On the local embedder the current numbers
are **hit-rate@4 = 1.0 and MRR = 0.635** across 8 cases, and it's exactly what let
me catch that turning on the local reranker *lowered* MRR to 0.521, and back the
default out. Run it before and after a change and compare. (With the fake embedder
the numbers are only a plumbing check; point it at the local or hosted embedder
for meaningful scores.)

---

## Voice-to-transcript

This is the bonus. The browser records and transcribes locally through the Web
Speech API (`streamlit-mic-recorder`), and only the recognised text is sent to the
backend, where it runs through the same pipeline a file would. The one honest
caveat is that browser STT has **no diarisation**: it can't tell speakers apart,
so voice input is attributed to a single speaker with synthesised timestamps.

Real diarisation ("who spoke when") needs the raw waveform and a speaker-embedding
model (VAD, then x-vectors, then clustering, e.g. pyannote), which isn't practical
client-side. Rather than build a browser DSP stage, I treat diarisation as a
property of the transcription backend: the `Transcriber` seam lets a server-side
provider (Deepgram, AssemblyAI, Whisper+pyannote) that returns diarised turns drop
in without touching anything downstream. So the upgrade isn't "nicer text", it's
"voice finally produces the same first-class speaker turns a file gives you".

---

## Productionising this

What I'd need to deploy this on a hyperscaler and let it scale:

- **Stateless API behind autoscaling** (ECS / Cloud Run / GKE). The app already
  builds its services from config, so horizontal scaling is just more replicas.
- **Managed vector store**: move Chroma to pgvector (RDS / Cloud SQL) or a managed
  Qdrant / Pinecone. The `VectorStore` interface already isolates this.
- **Async ingestion**: push embedding off the request path onto a queue (SQS plus
  workers) so large transcripts don't block. Ingestion is idempotent, so retries
  are safe.
- **Secrets** in a manager (Secrets Manager / KMS), never env files in prod.
- **AuthN/Z and tenancy**: meetings scoped per user or org. Today `meeting_id`
  filtering is the only isolation.
- **Observability**: ship the structured logs to CloudWatch or Datadog, add
  tracing (OpenTelemetry) across the retrieve/rerank/generate stages, and alert on
  grounded-rate and latency.
- **Caching**: cache embeddings for repeated ingests, and optionally answers.
- **CI/CD**: run `ruff`, `pytest`, and the eval harness on every PR, with a
  regression gate on hit-rate and MRR.

---

## Engineering standards followed (and skipped)

**Followed:** typed throughout (pydantic plus type hints); clear layering with
dependency-inverted seams; config over constants; idempotent ingestion; unit tests
on the deterministic logic (parser, redactor, chunker) plus end-to-end API tests,
all runnable with zero keys; structured logging; a Dockerfile (non-root, with a
healthcheck) and compose; a ruff config; and a Makefile for the common flows.

**Skipped, deliberately and with reasons:** no auth or multi-tenancy (out of scope
for the exercise); PII detection is regex-based, so it misses names-in-context, and
Presidio is wired as an optional upgrade rather than the default; the eval set is
small and hand-made, not a statistically meaningful benchmark; there's no
retry/backoff around the hosted APIs yet; and the UI is Streamlit, since the brief
said UI polish isn't the bar, so I chose speed over polish and treated the API as
the real product surface.

---

## How I used AI tools

I used an AI assistant as a **design partner and a fast typist**, not an author.
The way it worked: I drove the architecture decisions in conversation. The
seam-based design, the per-turn chunking, the redaction call (keep names, redact
contact PII), treating rerank as a cross-encoder, and putting diarisation in the
backend are all mine, and several were places where I *rejected* a
plausible-sounding suggestion (translate-to-English embedding, stopword removal,
browser-side FFT diarisation) because it was subtly wrong. The AI earned its keep
on scaffolding (pydantic models, the Chroma adapter, test skeletons) and on
pressure-testing my reasoning.

My do's and don'ts, learned the hard way here: **do** use it to draft the
mechanical parts and to argue against your own design; **don't** let it make the
judgment calls or write the README's reasoning, because that has to be yours, or
the whole point (showing how *you* think) is lost. Everything it produced I read,
edited, and can defend line by line. This README is written by me, about my own
decisions.

---

## What I'd do differently / add next

Part of the original "next" list is already **done**: action-item and decision
extraction, conversational memory, the whole-meeting brief, and the reranker
(wired and measured, then left off because the eval said so). What I'd take on
next, roughly in priority order:

1. **Hybrid retrieval** (dense plus BM25). Meetings are full of exact terms
   (names, dates, project codenames, dollar figures) that keyword search nails and
   pure semantic search sometimes blurs. Given the reranker result above, this is
   now my best bet for retrieval quality on this corpus.
2. **LLM-based extraction** behind the existing `ExtractedItem` seam, to replace
   the cue-phrase baseline and catch phrasings the regex misses, plus an LLM
   narrative summary added to the brief at ingestion.
3. **Grow the eval set** (it's 8 hand-made cases today) and add an *answer*-quality
   metric (faithfulness and citation-correctness), not just retrieval hit-rate and
   MRR, then gate CI on regressions. I'd also add an eval case type for aggregation
   recall.
4. **A domain-tuned or larger reranker**, then re-measure. The mechanism is right
   even if the small model wasn't.
5. **Presidio** as the default redactor for names-in-context.
6. **Streaming responses** and answer caching.
7. **Server-side diarised transcription** behind the `Transcriber` seam.
