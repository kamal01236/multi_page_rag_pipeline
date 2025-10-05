# Low-Level Design (LLD) — Multi-Page RAG Pipeline

This LLD specifies an authoritative, implementable design for a Multi-Page Retrieval-Augmented Generation (RAG) pipeline that supports N pages, per-page QA, and safe RetrievalQA behavior. It defines the data model, required public functions and signatures, pseudocode for core operations, incremental ingestion and FAISS merge guidance, confidence heuristics, logging contracts, and test cases.

Primary goals
- Index N pages into a single persisted vector DB (FAISS) with per-chunk provenance metadata.
- Provide deterministic TF-IDF fallback for CI/offline tests.
- Expose a small HTTP API for ingestion and QA.
- Provide clear, environment-first configuration and separation of concerns.

References to implemented code (open to inspect wiring)
- Configuration: [`Config`](src/config.py), [`load_config`](src/config.py), [`CONFIG`](src/config.py) — [src/config.py](src/config.py)
- Vectorstore & search: [`build_vectorstore`](src/rag_pipeline.py), [`search_db`](src/rag_pipeline.py), [`SimpleFallbackVectorStore`](src/rag_pipeline.py) — [src/rag_pipeline.py](src/rag_pipeline.py)
- RetrievalQA, LLM selection: [`build_retrieval_qa`](src/rag_pipeline.py), [`detect_backend`](src/rag_pipeline.py) — [src/rag_pipeline.py](src/rag_pipeline.py)
- Page-specific askers factory: [`make_page_specific_askers`](src/rag_pipeline.py) — [src/rag_pipeline.py](src/rag_pipeline.py)
- CLI integration: [src/cli.py](src/cli.py)
- HTTP endpoints: [src/server_api.py](src/server_api.py)

---

1. From requirements to design to implementation (overview)
- Requirements
  - Fetch and parse N web pages (Wikipedia used in demo).
  - Clean text to remove menus/citations, produce readable chunks.
  - Randomized overlapping chunking (400–600 chars, 50-char overlap, seedable).
  - Store chunks in a vector DB with per-chunk metadata (source URL, title, chunk_index).
  - Similarity search with post-retrieval source filtering.
  - RetrievalQA chain that uses only retrieved chunks, returns "I don't know based on the provided documents." when evidence insufficient, and includes a `Sources:` section with URLs.
  - Per-page askers (page-specific QA) that are dynamic and not hard-coded.

- Design choices (implemented)
  - Centralized, environment-first configuration: see [`src/config.py`](src/config.py) — env overrides config file; keys: RAG_DEFAULT_BACKEND, RAG_EMB_BACKEND, RAG_LLM_BACKEND, FORCE_OLLAMA, OPENAI_API_KEY, OLLAMA_URL, OLLAMA_TOKEN, HUGGINGFACEHUB_API_TOKEN.
  - Separation of concerns:
    - config loader: [`src/config.py`](src/config.py)
    - ingestion & chunking: [`src/rag_pipeline.py::build_from_urls`](src/rag_pipeline.py)
    - embeddings/vectorstore: [`src/rag_pipeline.py::build_vectorstore`](src/rag_pipeline.py)
    - retrieval & LLM QA: [`src/rag_pipeline.py::build_retrieval_qa`](src/rag_pipeline.py)
    - HTTP surface: [`src/server_api.py`](src/server_api.py)
  - Deterministic fallback: `SimpleFallbackVectorStore` (TF-IDF) in [`src/rag_pipeline.py`](src/rag_pipeline.py) for CI/test reproducibility.

- Implementation highlights
  - Chunk metadata includes `source`, `title`, `chunk_index`. Stored vector ids and metadata mapping strategy is documented in this LLD and implemented as in-memory persistence for the demo; production should persist FAISS + metadata JSONL (see Incremental Ingestion).
  - `search_db` performs candidate fetching using a multiplier (k * 3) and applies post-retrieval filtering by source substring match (case-insensitive).
  - `build_retrieval_qa` wraps a strict prompt that instructs the LLM not to hallucinate and to include a `Sources:` section; if confidence is low (configurable), returns the canonical "I don't know..." string.
  - `make_page_specific_askers` is dynamic: it enumerates indexed sources and returns slug → asker functions that call the QA callable with `filter_source`.

---

2. API Endpoints (detailed) — implemented in [`src/server_api.py`](src/server_api.py)

All endpoints are defined in [src/server_api.py](src/server_api.py).

- GET /healthz
  - Purpose: basic liveness check and current backend.
  - Response: { "status": "ok", "backend": "<backend>" }
  - Implementation: returns STATE["backend"].

- POST /ingest
  - Purpose: ingest one or more URLs and build vector store + QA chain + page askers.
  - Request model: IngestRequest { urls: List[str], seed?: int, backend?: str = "auto", overwrite?: bool }
  - Behavior:
    - Calls [`build_from_urls`](src/rag_pipeline.py) to fetch + clean + chunk + embed using the selected embedding backend.
    - Persists in-memory state (docs, vector_store).
    - Calls [`build_retrieval_qa`](src/rag_pipeline.py) to build a QA callable (LLM or TF-IDF fallback).
    - Calls [`make_page_specific_askers`](src/rag_pipeline.py) to construct page-specific askers.
    - Returns JSON with pages_indexed, total_chunks, backend, askers_available.
  - Errors:
    - 500 if ingestion or QA chain construction fails.

- GET /pages
  - Purpose: list indexed unique source URLs.
  - Response: { count: int, pages: [urls...] }

- POST /qa
  - Purpose: general QA across all indexed pages.
  - Request model: QARequest { query: str, k?: int, filter_source?: Optional[str] }
  - Behavior:
    - If a QA callable exists (LLM or simple), calls it and returns its answer + sources.
    - Otherwise, falls back to [`search_db`](src/rag_pipeline.py) to return concatenated chunks plus confidence.
  - Errors:
    - 400 if no vector store present (ingest first).
    - 500 on server errors.

- POST /qa/{slug}
  - Purpose: page-specific QA using slug created from source URL.
  - Behavior:
    - Find created askers via STATE["askers"] and invoke the slug function.
    - Returns answer + sources or 404-like message if slug not found.

- POST /batch_qa
  - Purpose: run multiple queries in a batch.
  - Request model: BatchQARequest { queries: List[str], k?: int }
  - Behavior:
    - Iterates queries, calls /qa logic for each, returns results list.
  - Note: For heavy workloads use async/streaming or queue-based processing in production.

Mapping endpoints → code
- Ingestion: [`build_from_urls`](src/rag_pipeline.py) → [`build_vectorstore`](src/rag_pipeline.py) → persisted STATE in [src/server_api.py](src/server_api.py).
- QA: [`build_retrieval_qa`](src/rag_pipeline.py) produces callable used by /qa and page askers.
- Post-filtering: [`search_db`](src/rag_pipeline.py) implements post-retrieval filtering.

---

3. Data model & persistence (short)
- Chunk representation (in memory): Document (or dict) with fields: page_content, metadata: { source, title, chunk_index, ingest_ts }.
- Production persistence:
  - FAISS index files on disk (or cloud object store).
  - Metadata JSONL mapping id → metadata (atomic write via .tmp → rename).
  - Deterministic mapping between FAISS vector ids and metadata ids is required (store explicit id mapping or use consistent sequential IDs).

---

4. Confidence & hallucination guard (implemented)
- Compute normalized confidence from retrieved scores (map to [0,1]). Default threshold recommended 0.35 (see config).
- If confidence < threshold, return exactly: "I don't know based on the provided documents." and do not call LLM.
- The LLM prompt contains the hard constraint to only use provided context and include Sources.

---

5. Per-page askers
- [`make_page_specific_askers`](src/rag_pipeline.py) enumerates sources and generates slug keys (sanitized last path element).
- Each asker calls the QA callable with `filter_source` set to that canonical URL (post-retrieval filtering).

---

6. Incremental ingest & index update (guidance)
- Append new vectors using FAISS.add(new_vectors) with consistent id mapping.
- Append metadata atomically to metadata JSONL; when overwriting use `--overwrite` semantics: remove previous chunks for the same source then add new ones.
- Use file locks for concurrency when writing FAISS + metadata.

---

7. Observability & logs (what is emitted)
- search_db: INFO logs with query, k_requested, k_returned, filter_source.
- QA invocation: INFO logs with query, confidence, used_sources, llm_backend. WARNING on missing citations or fallback.
- Ingest: INFO logs with urls_processed, chunks_created, embeddings_backend.

---

8. Tests (what to run)
- Unit tests should rely on TF-IDF fallback for determinism.
- Integration tests use the HTTP endpoints in [src/server_api.py](src/server_api.py) with monkeypatched pipeline internals to avoid external network calls.
- Provided test file: tests/test_server_endpoints_full.py (see tests/) covers:
  - /healthz
  - /ingest success and failure
  - /pages listing
  - /qa with qa_callable and fallback
  - /qa/{slug} page-specific behavior
  - /batch_qa

---

9. Failures and mitigation
- External APIs (OpenAI) may fail or rate-limit — system falls back to HuggingFace or TF-IDF.
- If LLM fails at runtime, the server will fallback to the simple retrieval answer.
- For production robustness, add retry/backoff, circuit-breaker, and metrics.

---

10. Next production hardening recommendations
- Persist FAISS + metadata to durable storage and implement safe index migration/compaction.
- Add authentication + rate-limiting to HTTP endpoints.
- Add monitoring and metrics (Prometheus + logs).
- Use secrets manager for API keys (do not store keys in config file).
- Add CI job that runs TF-IDF-only integration tests to avoid dependence on cloud APIs.

---

Appendix: Quick pointer to code (open these)
- [`src/config.py`](src/config.py) — configuration loader (`CONFIG`)
- [`src/rag_pipeline.py`](src/rag_pipeline.py) — chunking, `build_vectorstore`, `search_db`, `build_retrieval_qa`, `make_page_specific_askers`, `build_from_urls`
- [`src/server_api.py`](src/server_api.py) — HTTP endpoints implementation
- [`src/cli.py`](src/cli.py) — CLI example and ingestion flow

*** End of LLD.md