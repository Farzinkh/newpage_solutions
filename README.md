# Meeting Intelligence

A conversational assistant that answers questions about meeting transcripts —
decisions, action items, "what did we say about X" — with every answer grounded
in the transcript and cited back to the **speaker, meeting, and timestamp**.
Handles questions that span several meetings, and supports both uploaded
transcripts and browser voice-to-text.

This is **Option 3** (Meeting Intelligence) of the assignment. The code lives in
[`meeting-intelligence/`](meeting-intelligence/).

---

## Quick start (Docker)

The zero-key demo runs with fake backends (fake embedder, in-memory store, echo
LLM) — **no API keys, no model downloads** — and auto-loads three sample
meetings. It's a lightweight image, so it builds in seconds.

```bash
cd meeting-intelligence
docker compose -f docker-compose.demo.yml up --build
```

Then open the UI: **http://localhost:8501**
(API is on http://localhost:8000 — try http://localhost:8000/health)

> If 8000/8501 are already in use, remap the host ports in
> `docker-compose.demo.yml` (e.g. `18000:8000`).

In the UI, click one of the **example prompts** (e.g. *"List all the action
items across every meeting"*) and the grounded, cited answer appears. The
answer text is extractive in this profile — the echo LLM doesn't reason — but
retrieval, citations with absolute timestamps, the per-meeting brief, and the
two-stage cross-meeting grouping are all real and inspectable.

### Real answers (local embeddings + a hosted LLM)

For genuine semantic retrieval and synthesized answers, use the full image and
one API key:

```bash
cd meeting-intelligence
OPENAI_API_KEY=sk-...  EMBEDDER_BACKEND=local VECTOR_STORE_BACKEND=chroma \
  LLM_BACKEND=openai  docker compose up --build
# UI on http://localhost:8501, API on http://localhost:8000
```

### Without Docker

See [`meeting-intelligence/README.md`](meeting-intelligence/README.md#quick-start)
for the local Python setup, tests, and the retrieval eval harness.

---

## Documentation

- **[`meeting-intelligence/README.md`](meeting-intelligence/README.md)** — the
  main write-up: RAG/LLM decisions and trade-offs, chunking, embeddings,
  retrieval (two-stage cross-meeting), guardrails, observability, what I'd
  productionise, engineering standards, and how AI tools were used. **Start here
  for the reasoning.**
- **[`architecture.md`](architecture.md)** — architecture diagrams (Mermaid):
  the runtime phases, dual-input flow, layer boundaries, domain model, and the
  lifecycle of a query.
- **[`meeting-intelligence/docs/architecture.md`](meeting-intelligence/docs/architecture.md)**
  — the same diagrams kept alongside the code.

---

## What it does (at a glance)

- **Ingests** transcripts (`[HH:MM:SS] Speaker: text` or plain `Speaker: text`)
  or browser voice input; **redacts** leaked PII before storage.
- **Anchors** each turn to an absolute datetime (`meeting_start + offset`) so
  citations are unambiguous across meetings.
- **Extracts** decisions and action items at ingestion, so "list the action
  items" / "summarise" are answered by enumeration, not top-k search.
- **Two-stage retrieval**: picks the meeting(s) that matter (injecting each one's
  brief), then digs the exact hits with their before/after context.
- **Refuses to hallucinate** and **shows its work** (retrieved chunks + scores).

See the [main README](meeting-intelligence/README.md) for the full picture.
