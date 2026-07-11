# Meeting Intelligence

A conversational assistant that answers questions about meeting transcripts —
decisions, action items, "what did we say about X" — with answers grounded in
the transcript and cited back to the speaker, meeting, and (absolute) timestamp.
Handles questions that span several meetings, and supports both uploaded
transcript files and browser voice-to-text.

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

Docker:
- **Zero-key demo** (fake backends, auto-seeded, lightweight image):
  `docker compose -f docker-compose.demo.yml up --build` → UI on
  **http://localhost:8501**, API on `:8000`. Click an example prompt to see a
  cited answer.
- **Real profile** (local embeddings + Chroma + a hosted LLM):
  `docker compose up --build` → UI on `:8501`, API on `:8000`.

---

## What it does

- **Ingests** transcripts (`[HH:MM:SS] Speaker: text`, or plain `Speaker: text`)
  or browser voice input.
- **Redacts** leaked PII (emails, phones, SSNs, card numbers) before storage.
- **Extracts** decisions and action items at ingestion, so "list the action
  items" / "summarise the meeting" are answered by enumeration, not top-k search.
- **Retrieves** relevant turns and **generates** an answer that cites its sources,
  with the **whole-meeting brief** (participants, decisions, action items)
  injected as context so answers aren't limited to isolated fragments.
- **Is conversational**: prior turns are fed back and follow-ups ("who owns
  that?") are rewritten to standalone queries before retrieval.
- **Refuses to hallucinate**: if the transcript doesn't cover it, it says so.
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
sources converge to `Turns` at the transcriber seam, so everything after it is
shared. Retrieval is two-stage (pick the meetings that matter, then dig each with
neighbour context); reranking is wired but off by default (the eval showed it
didn't help — see below). See [`docs/architecture.md`](docs/architecture.md) for
the fuller Mermaid diagrams — the dual-input flow, layer dependencies, domain
model, and the two-stage query lifecycle.

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

**Retrieval — wide net, then (optional) rerank, and a measurement that changed my
mind.** The embedder is a *bi-encoder*: it encodes the query and each chunk
separately, so it never compares them directly and misses fine-grained relevance.
The textbook fix is retrieve top-N (=20) cheaply by cosine similarity, then rerank
with a *cross-encoder* (scores each query–chunk pair jointly) down to the top-k
(=4) the LLM sees. I wired **two** rerankers — a keyless local cross-encoder
(`ms-marco-MiniLM`) and hosted **Cohere Rerank** — intending to turn reranking on
by default.

Then I ran the eval, and it *overruled the intuition*: on this corpus the local
cross-encoder made retrieval **worse** (hit-rate@4 1.0 → 0.875, MRR 0.635 →
0.521 — it demoted the correct chunk for one query below rank 4). The small
MS-MARCO model is trained on web-passage ranking; short conversational meeting
turns are out of its distribution, and the multilingual bi-encoder already nails
top-4 recall here. So the honest call is: reranking stays **wired, tested, and
behind a flag** (`RERANKER_BACKEND`), but **off by default**, because the numbers
say it doesn't help *this* data. This is exactly what the eval harness is for —
turning a plausible default into a measured one. I'd revisit it with a larger,
domain-tuned reranker and a bigger eval set. Both scores are logged and shown in
the UI.

**Orchestration — no framework.** I wrote the ~150 lines of RAG glue directly
rather than pulling in LangChain/LlamaIndex. At this scope a framework's
abstractions cost more legibility than they save, and hand-rolled code is easier
for a reviewer to read and for me to reason about. The interfaces give me the
composability a framework would, without the indirection. (For a much larger
system with many loaders and tools, that calculus flips.)

**Beyond top-k — the three things that make it a *meeting* assistant, not a
generic RAG box.** Plain chunk retrieval has three failure modes that matter a
lot for meetings specifically, and each has its own mechanism:

1. **Aggregation queries are extraction, not retrieval.** "What are all the
   action items?" needs *every* relevant turn; top-k returns a handful (the eval
   shows a 3-turn action-item answer retrieving 1 of 3). So a deterministic,
   keyless pass at ingestion tags decisions and action items (high-precision cue
   phrases — I tuned these toward precision, dropping broad cues like bare
   "let's" that tagged "let's wrap" as a to-do). An intent detector routes
   enumerations/summaries to answer from those records, so the list is *complete*
   and every line is cited. The documented upgrade is an LLM extraction pass
   behind the same `ExtractedItem` seam.

2. **Isolated chunks lose the meeting's narrative — so retrieval is two-stage.**
   A turn retrieved on its own doesn't know the meeting decided single-device-only
   or that Daniel owns the doc, and a flat top-k across meetings can't tell which
   meeting to trust. So the query path is **coarse-to-fine** (`retrieval/planner.py`):
   - *Stage 1 (which meetings):* group the wide-net candidates by meeting, rank
     meetings by their best hit, keep the top few (`MAX_MEETINGS`). Each selected
     meeting contributes its **brief** — participants, decisions, action items,
     derived once at ingestion — as clearly-marked, *non-citable* orientation.
   - *Stage 2 (dig in):* for each meeting take its top hits (`PER_MEETING_HITS`)
     and expand each with `CONTEXT_WINDOW` turns **before and after** (a metadata
     lookup by `turn_index`, not a second similarity search), so the model reads
     each quote *in the flow of the conversation*.

   The answer prompt is then grouped by meeting: each block is an overview plus
   its excerpts (hits and their `(context)` neighbours), every turn numbered and
   citable. So a question spanning several meetings gets both the whole-meeting
   shape of *each* relevant meeting and the local context around each hit —
   instead of four orphaned fragments. An optional first LLM pass
   (`TWO_PASS_REASONING`) reads the briefs and forms an early hypothesis of which
   meeting matters before the answer pass. (`/brief` exposes a brief directly.)

3. **A one-shot Q&A box isn't "conversational."** The brief asks for a
   *conversational* system, so `/query` takes prior turns. A follow-up like "who
   owns that?" is first **rewritten to a standalone question** (using a real LLM;
   for the keyless EchoLLM it degrades to carrying the previous question's terms)
   so retrieval has the referent, and the recent turns are also passed to the
   answer prompt to resolve references. Capped at `MAX_HISTORY_TURNS`.

4. **Cross-meeting questions need attribution and absolute time.** A question can
   pull turns from several meetings at once (search is unscoped by default). Two
   things make that coherent instead of confusing: (a) every excerpt, citation,
   and extracted item is **labelled with its meeting**; and (b) relative
   transcript timestamps (`00:02:36`, which collide across meetings) are anchored
   to **wall-clock datetimes** at ingestion — `meeting_start + offset`, where
   `meeting_start` comes from the file name (`name_2026-06-03_1548.txt`) or an
   explicit `started_at`. So `[2] (incident_retro · Omar @ 2026-06-05 10:15:33)`
   is globally unambiguous. This is a strictly additive preprocessing step: with
   no date in the name it falls back to the relative timestamp. (The parser also
   strips the date/time back off to recover a stable `meeting_id`, so naming
   conventions don't leak into identifiers or the eval.)

**Prompt & context management.** The LLM gets, in order: the recent conversation
(if any), then the retrieved context **grouped by meeting** — each meeting block
is its non-citable brief followed by numbered excerpts, each prefixed with
`[n] (meeting · speaker @ time)` and marked `(context)` when it's a
before/after neighbour rather than a hit — and a system prompt that says: answer
only from the numbered excerpts, cite as `[n]`, never cite the brief or history,
and if the answer isn't there say exactly "Not discussed in the transcript."
`temperature=0` for determinism.

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

**Observability.** Every query logs one JSON line: the question, the route taken
(aggregation vs. retrieval, and how many meetings it spanned), retrieved chunk
ids, similarity and rerank scores, grounded flag, and per-stage latency
(embed / search / llm / total). The UI surfaces the same scores under each answer.
Only redaction *counts* are logged, never raw PII values.

**Quality — a real eval harness.** `eval/gold_set.json` maps hand-curated
questions to the turns that answer them; `eval/evaluate.py` computes hit-rate@k
and MRR. This is what makes every knob (k, rerank on/off, embedding model) a
*measured* decision instead of a vibe. On the local embedder the current numbers
are **hit-rate@4 = 1.0, MRR = 0.635** (8 cases); this is what let me catch that
turning on the local reranker *lowered* MRR to 0.521 and back the default out.
Run it before and after a change and compare. (With the fake embedder the numbers
are only a plumbing check — point it at the local or hosted embedder for
meaningful scores.)

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

Some of the original "next" list is now **done** — action-item/decision
extraction, conversational memory, the whole-meeting brief, and the reranker
(wired + measured, then left off because the eval said so). What I'd tackle next,
in rough priority order:

1. **Add hybrid retrieval** (dense + BM25). Meetings are full of exact terms
   (names, dates, project codenames, dollar figures) that keyword search nails
   and pure semantic search sometimes fuzzes. Given the reranker result above,
   this is now my top retrieval-quality bet for this corpus.
2. **LLM-based extraction** behind the existing `ExtractedItem` seam, replacing
   the cue-phrase baseline for recall on phrasings the regex misses — and
   extend the brief with an LLM narrative summary at ingestion.
3. **Grow the eval set** (it's 8 hand-made cases) and add an *answer*-quality
   metric (faithfulness/citation-correctness), not just retrieval hit-rate/MRR;
   gate CI on regressions. Also add an eval case type for aggregation recall.
4. **A domain-tuned / larger reranker** and re-measure — the mechanism is right
   even if the small model wasn't.
5. **Presidio** as the default redactor for names-in-context.
6. **Streaming responses** and answer caching.
7. **Server-side diarised transcription** backend behind the `Transcriber` seam.
