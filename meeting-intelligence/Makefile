.PHONY: install dev api ui test eval lint seed demo

install:
	pip install -e ".[dev]"

dev:
	pip install -e ".[dev,rerank,openai,anthropic,ui,pii]"

api:
	uvicorn app.api:app --reload --port 8000

ui:
	streamlit run ui/streamlit_app.py

test:
	pytest -q

lint:
	ruff check .

eval:
	python -m eval.evaluate

# Ingest the sample transcripts into a running backend.
seed:
	python scripts/seed.py

# Run everything with zero API keys (fakes) — handy for a quick smoke test.
demo:
	EMBEDDER_BACKEND=fake VECTOR_STORE_BACKEND=memory RERANKER_BACKEND=noop LLM_BACKEND=echo \
	  uvicorn app.api:app --port 8000
