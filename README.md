# Multi-Page RAG Pipeline — README (updated)

This project implements a Retrieval-Augmented Generation (RAG) pipeline that indexes multiple web pages, supports per-page QA, and provides a small HTTP API for ingestion and queries. The code follows industry-standard configuration and separation of concerns.

Key code locations
- Config loader: [`src/config.py`](src/config.py) — use `CONFIG` env-first semantics.
- Core pipeline: [`src/rag_pipeline.py`](src/rag_pipeline.py) — chunking, vectorstore, retrieval, QA.
- Server API: [`src/server_api.py`](src/server_api.py) — endpoints for ingest/qa/pages.
- CLI: [`src/cli.py`](src/cli.py) — example ingestion/interactive usage.

Features
- Randomized overlapping chunking (400–600 chars, 50 overlap).
- Per-chunk metadata with source URL and chunk index.
- Multi-backend support: OpenAI (remote), Ollama (local), HuggingFace (local/hosted), TF-IDF fallback (deterministic).
- RetrievalQA that returns only evidence-backed answers and includes `Sources:` in responses.
- Page-specific askers that restrict answers to a single source (dynamic, not hard-coded).
- FastAPI endpoints for ingestion and QA.

Endpoints (examples)
- GET /healthz — liveness and current backend
- POST /ingest — { urls: [..], seed?: int, backend?: "auto"|... } → builds index + QA callable + askers
- GET /pages — list indexed pages
- POST /qa — { query: str, k?: int, filter_source?: str } → returns answer, sources, backend, confidence
- POST /qa/{slug} — page-specific QA by slug
- POST /batch_qa — run multiple queries in batch

How it works (high level)
1. Ingest flow: user supplies URLs → [`build_from_urls`](src/rag_pipeline.py) fetches and cleans HTML → `chunk_text` chunker creates randomized overlapping chunks → [`build_vectorstore`](src/rag_pipeline.py) creates FAISS or TF-IDF vectorstore storing per-chunk metadata.
2. QA flow: incoming query → [`search_db`](src/rag_pipeline.py) retrieves top candidates (k * multiplier) → apply post-retrieval filtering (filter_source) → compute confidence → if enough evidence call LLM through [`build_retrieval_qa`](src/rag_pipeline.py) else return "I don't know based on the provided documents.".
3. Page askers: [`make_page_specific_askers`](src/rag_pipeline.py) produces slug→callable that invokes QA with `filter_source`.

Configuration
- Environment-first loader in [`src/config.py`](src/config.py)
- Set secrets via env: `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, `OLLAMA_TOKEN`.
- Optional config file at `./config/config.yaml` or path set by `RAG_CONFIG_FILE` (env).
- Control backends via `RAG_EMB_BACKEND`, `RAG_LLM_BACKEND`, `RAG_DEFAULT_BACKEND`, and `FORCE_OLLAMA`.

Quickstart
1. Install:
   pip install -r requirements.txt

2. Run demo using TF-IDF fallback (no keys required):
   python -m src.cli --urls https://en.wikipedia.org/wiki/Quantum_computing https://en.wikipedia.org/wiki/Quantum_machine_learning --backend tfidf

3. Run server:
   uvicorn src.server_api:app --reload

4. Example API usage:
   POST /ingest with JSON { "urls": ["https://.../Quantum_computing", "https://.../Quantum_machine_learning"] }
   POST /qa with JSON { "query": "What is a qubit?" }

Testing
- Unit & integration tests rely on TF-IDF fallback for deterministic results.
- Run tests:
   pytest -q

- New comprehensive endpoint tests: `tests/test_server_endpoints_full.py`

Deliverables
- Source: `src/` (pipeline and server)
- Notebook: `notebooks/multi_page_rag_notebook.ipynb`
- LLD: `LLD.md` (this file)
- README: `README.md` (this file)
- Tests: `tests/` (including endpoint tests)
- Dockerfile & docker-compose for local demonstration (ollama placeholder)

Production recommendations
- Persist FAISS + metadata JSONL and provide migration utilities.
- Use secrets manager for API keys.
- Add authentication, rate limiting, observability (metrics/tracing).
- Implement file locks for index writes and safe FAISS merges for incremental ingestion.

References
- [`src/config.py`](src/config.py)
- [`src/rag_pipeline.py`](src/rag_pipeline.py)
- [`src/server_api.py`](src/server_api.py)
- [`src/cli.py`](src/cli.py)

Support
- For additions (OpenAPI, CI, or persistent storage), open an issue or request a patch that updates `src/server_api.py`, `src/rag_pipeline.py`, and adds migration tests.

---

End-to-end implementation summary

This repository implements the full pipeline from requirement → design → implementation:

- Requirement: fetch two or more web pages, clean, chunk, embed, and answer queries with per-page filtering and citation.
- Design: clear separation of ingestion (fetch/clean/chunk), vectorstore (embedding selection / FAISS), retrieval (`search_db`), QA (`build_retrieval_qa`) and an HTTP surface (`src/server_api.py`).
- Implementation:
   - `src/rag_pipeline.py` contains the ingestion helpers, `build_from_urls`, `chunk_text`, `build_vectorstore`, `search_db`, `build_retrieval_qa` and `make_page_specific_askers`.
   - `src/cli.py` demonstrates ingest + interactive query usage.
   - `src/server_api.py` provides REST endpoints for ingestion and QA.
   - `notebooks/multi_page_rag_notebook.ipynb` contains an executable demo of N-page ingestion and per-page askers.

All deliverables are included: code, notebook, tests, LLD, README, and Docker templates. If you'd like, I can:
- Add an OpenAPI schema for the FastAPI app and include it in the README.
- Add CI workflows that run TF-IDF-only tests to avoid external API keys in CI.


