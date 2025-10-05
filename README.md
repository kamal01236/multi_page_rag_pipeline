# Multi-Page RAG Pipeline
Repository: kamal01236/heart_attack_prediction

This project implements a Retrieval-Augmented Generation (RAG) pipeline that indexes multiple web pages, supports per-page QA, and provides a small HTTP API for ingestion and queries. The code follows industry-standard configuration and separation of concerns.
# Multi-Page RAG Pipeline — Single authoritative README

This file explains the approach, architecture, implementation details, and step-by-step instructions to run the RAG pipeline API locally.

Contents
- Overview and approach
- Architecture & components
- Data model & persistence
- Key algorithms and decisions (chunking, TF‑IDF fallback, retrieval filtering)
- HTTP API (endpoints and behavior)
- How to run locally (PowerShell-friendly steps)
- Tests and CI guidance
- Observability, failure modes, and production notes
- Next steps and recommendations

Deliverables

- Implementation: core pipeline and API (see `src/rag_pipeline.py`, `src/server_api.py`, `src/cli.py`).
- Deterministic TF‑IDF fallback for CI/tests: `SimpleFallbackVectorStore` in `src/rag_pipeline.py` (uses scikit-learn `TfidfVectorizer`).
- Page-specific askers and per-page QA: `make_page_specific_askers` in `src/rag_pipeline.py` and server endpoints `/pages` and `/qa/{slug}` in `src/server_api.py`.
- Notebook demo and verification: `notebooks/multi_page_rag_notebook.ipynb` (demo + test-like checks) and `notebook_run_checks.py` (script to reproduce checks).
- Tests: unit tests under `tests/` that use the TF‑IDF fallback for deterministic results.
- Documentation and implementation notes: `README.md` (this file) and `IMPLEMENTATION.md` (detailed summary of decisions and steps taken).

Goal: provide a deterministic, testable Multi-Page RAG pipeline that can ingest N web pages, split them into randomized overlapping chunks with provenance metadata, build a vector store (FAISS when available, deterministic TF‑IDF fallback for CI), and answer queries either across all pages or restricted to a specific page via dynamic per-page askers.

Approach taken
- Environment-first configuration: runtime choices (embedding backend, LLM backend, API keys) are read from environment variables with an optional YAML config file fallback.
- Separation of concerns: ingestion/chunking, vectorstore construction, retrieval, QA chain construction, and HTTP surface are separate modules.
- Deterministic fallback: a TF‑IDF vector store is provided for CI/offline runs so unit tests do not require external API keys or network.
- Post-retrieval filtering: search returns candidate chunks and filtering by source is applied after retrieval for robust per-page askers.

Why this design
- Makes tests/CI reliable by avoiding external dependencies.
- Supports production-quality backends (OpenAI, HuggingFace, Ollama) while preserving a local fallback.
- Keeps answers traceable via per-chunk metadata and `Sources:` in responses.

---

Architecture & components

Folder/key files
- `src/config.py` — env-first configuration loader and `CONFIG` object.
- `src/rag_pipeline.py` — core pipeline: fetching/cleaning, chunking, build_vectorstore, SimpleFallbackVectorStore (TF‑IDF), search_db, build_retrieval_qa, make_page_specific_askers, build_from_urls.
- `src/server_api.py` — FastAPI HTTP endpoints: `/healthz`, `/ingest`, `/pages`, `/qa`, `/qa/{slug}`, `/batch_qa`.
- `src/cli.py` — CLI demo to ingest and query from the terminal.
- `notebooks/multi_page_rag_notebook.ipynb` — executable demo and test-like checks.
- `tests/` — unit & endpoint tests that use the TF‑IDF fallback for determinism.

Runtime state
- The server maintains an in-memory STATE for `docs`, `vector_store`, `qa_callable`, `askers`, `metadata`, and `backend`. For production, persist FAISS and metadata.

---

Data model & persistence

Chunk representation (in-memory)
- Each chunk is represented as a Document-like object with:
   - `page_content` (text)
   - `metadata`: { source: URL, title: page title, chunk_index: int, ingest_ts: timestamp }

Persistence strategy for production
- Persist FAISS indexes (files) and a metadata JSONL mapping vector ids → metadata.
- Use atomic writes (write to .tmp then rename) and a file lock when updating indexes and metadata concurrently.
- Ensure deterministic id mapping between FAISS vector ids and metadata entries.

---

Key algorithms and decisions

Chunking
- Randomized overlapping chunking of cleaned text with parameters: min 400, max 600 chars, overlap 50 chars.
- Seedable randomness for deterministic tests. Implemented in `chunk_text`/`randomized_chunks`.

Vector store selection
- Backend priority: auto → OpenAI embeddings (via LangChain & FAISS) → HuggingFace → Ollama local embeddings → SimpleFallbackVectorStore (TF‑IDF).
- TF‑IDF fallback implemented in `SimpleFallbackVectorStore` using scikit-learn's TfidfVectorizer for deterministic behavior in CI.

search_db behavior
- Retrieves candidate_count = k * 3 (configurable multiplier), then applies post-retrieval filtering by metadata['source'] substring if requested.
- Returns top-k filtered hits with chunk text, score, and source.

RetrievalQA
- `build_retrieval_qa` tries to construct a LangChain RetrievalQA (LLM-backed). If unavailable or misconfigured, returns a simple fallback callable that concatenates top chunks and returns sources.
- The LLM prompt includes a strict instruction to only use provided context and append a `Sources:` list. The system returns "I don't know based on the provided documents." when evidence is insufficient.

Per-page askers
- `make_page_specific_askers` enumerates indexed sources and creates slug → callable mappings. Each asker passes `filter_source` to the QA callable or uses `search_db` when needed to ensure per-page restriction.

Confidence & hallucination guard
- Confidence computed from normalized retrieval scores. Default threshold (configurable) is recommended at ~0.35. If below threshold, pipeline returns the canonical refusal string.

---

HTTP API (endpoints & usage)

GET /healthz
- Response: { status: 'ok', backend: '<current backend>' }

POST /ingest
- Request JSON: { urls: [str], seed?: int, backend?: str (default 'auto'), overwrite?: bool }
- Behavior: fetches pages, cleans, chunks, builds vector store, constructs QA callable and page askers.
- Response: { pages_indexed: int, total_chunks: int, backend: str, askers_available: int }

GET /pages
- Response: { count: int, pages: [urls...] }

POST /qa
- Request: { query: str, k?: int, filter_source?: str }
- Response: { answer: str, sources: [str], backend: str, confidence: float }
- Error 400 when no vector store is present (ingest first).

POST /qa/{slug}
- Page-specific QA: looks up slug in askers and calls respective callable.

POST /batch_qa
- Request: { queries: [str], k?: int }
- Response: { count: int, results: [ { query, answer, sources } ] }

---

How to run the RAG pipeline API locally (PowerShell)

Prerequisites
- Python 3.11+.
- Recommended minimal packages for offline/demo mode: requests, beautifulsoup4, scikit-learn, numpy.

1) Create and activate virtual environment (PowerShell):

```powershell
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
```

2) Install dependencies (minimal / TF‑IDF-only):

```powershell
pip install requests beautifulsoup4 scikit-learn numpy
# For full feature set (LangChain/FAISS/OpenAI/HuggingFace):
pip install -r requirements.txt
```

3) Start the FastAPI server (from repo root):

```powershell
uvicorn src.server_api:app --reload
```

4) Ingest pages via the API (PowerShell example):

```powershell
#$body = '{\"urls\": [\"https://en.wikipedia.org/wiki/Quantum_computing\",\"https://en.wikipedia.org/wiki/Quantum_machine_learning\"]}'
#$resp = Invoke-RestMethod -Uri \"http://127.0.0.1:8000/ingest\" -Method POST -ContentType \"application/json\" -Body $body
```

5) Query the index (global QA):

```powershell
#$body = '{\"query\":\"What is a qubit?\"}'
#$resp = Invoke-RestMethod -Uri \"http://127.0.0.1:8000/qa\" -Method POST -ContentType \"application/json\" -Body $body
```

6) Page-specific QA (by slug):

```powershell
#$body = '{\"query\":\"What is a qubit?\"}'
#$resp = Invoke-RestMethod -Uri \"http://127.0.0.1:8000/qa/quantum_computing\" -Method POST -ContentType \"application/json\" -Body $body
```

7) Run notebook demo (optional):

```powershell
# install notebook deps if needed
pip install jupyter
jupyter notebook notebooks\\multi_page_rag_notebook.ipynb
```

Notes on backend selection
- If `OPENAI_API_KEY` is present and LangChain+FAISS are installed, the server will prefer OpenAI embeddings + FAISS. Otherwise it will try HuggingFace or Ollama, and finally TF‑IDF.

---

Tests and CI guidance

- Unit tests in `tests/` use the TF‑IDF fallback by default to ensure deterministic results in CI.
- Run tests locally:

```powershell
pip install pytest
pytest -q
```

- CI recommendation: install minimal runtime deps (scikit-learn, numpy) and run pytest; keep secrets out of CI.

---

Observability, failure modes & production notes

Logs & Observability
- `search_db`: INFO logs query and k results.
- `build_from_urls`: INFO logs fetch attempts and chunk counts.
- `build_retrieval_qa`: WARNING when falling back to simple QA.

Failure modes and mitigations
- External LLM failures: fallback to TF‑IDF/simple answers; add retry/backoff and circuit breaker in production.
- Index corruption during writes: use atomic metadata writes and file locks.

Security
- Use secret managers for API keys; do not commit keys to the repo.

---

Next steps & recommendations

- Persist FAISS + metadata and add migration/compaction utilities.
- Add an OpenAPI schema to `src/server_api.py` and surface it in the README.
- Add CI jobs that run TF‑IDF-only integration tests.
- Harden persistence with a proper file-locking mechanism and durable storage (S3/GCS for FAISS blobs and JSONL metadata).

---


