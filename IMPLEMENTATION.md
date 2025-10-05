Implementation Write-up — How the Multi-Page RAG problem was solved

This short write-up summarizes the concrete steps and engineering decisions used to implement the Multi-Page RAG pipeline in this repository.

Problem recap
- Ingest N web pages (Wikipedia in the demo), split into chunks with provenance metadata.
- Store chunk embeddings in a vector store and support similarity search with post-retrieval filtering by source.
- Provide a RetrievalQA layer that only uses retrieved evidence and returns Sources.
- Provide dynamic per-page askers that restrict answers to a single page.
- Provide deterministic, offline-capable behavior for CI and tests.

Key implementation steps

1) Config and environment-first design
- Implemented `src/config.py` which reads env vars and an optional config file. All backend selection and API key settings are env-first, making it easy to test locally and in CI.

2) Fetching and cleaning
- Implemented robust HTML fetching (`fetch_html`) and Wikipedia-specific cleaning (`clean_wikipedia_html`) to remove navigation, tables, citations and assemble readable paragraphs.

3) Deterministic randomized chunking
- Implemented seedable randomized overlapping chunking (min 400, max 600 chars, overlap 50). Seeded randomness lets tests be deterministic while preserving chunk variability for production runs.

4) Vector store + deterministic TF‑IDF fallback
- Implemented `build_vectorstore` to select embeddings & store implementation depending on environment:
  - Prefer production backends (OpenAI embeddings + FAISS via LangChain). 
  - Fall back to `SimpleFallbackVectorStore` which uses scikit-learn's TF‑IDF vectorizer for deterministic local tests.
- TF‑IDF fallback implements `search(query, k)` returning (doc, score) so tests and CI can run without API keys.

5) Post-retrieval filtering + `search_db`
- Implemented `search_db` to retrieve a candidate pool (k * 3) and then apply post-retrieval filtering by `metadata['source']` substring to produce final top-k hits. This enables per-page askers without rebuilding retrievers.

6) RetrievalQA and hallucination guard
- Implemented `build_retrieval_qa` to return either a LangChain RetrievalQA (if LLMs available) or a simple fallback callable that concatenates chunks and includes Sources.
- Added a confidence threshold derived from normalized retrieval scores. If below threshold, return: "I don't know based on the provided documents." This prevents hallucinations.

7) Dynamic per-page askers
- Implemented `make_page_specific_askers` to enumerate indexed sources and build slug→asker callables. Each asker performs a filtered retrieval (via `search_db` or by passing `filter_source` to the QA callable) and returns evidence + deduped sources.

8) FastAPI server + CLI + Notebook
- Implemented `src/server_api.py` with /ingest, /pages, /qa, /qa/{slug}, /batch_qa endpoints and a simple in-memory STATE for demo purposes.
- `src/cli.py` demonstrates ingestion and interactive querying from the terminal.
- A notebook (`notebooks/multi_page_rag_notebook.ipynb`) demonstrates an end-to-end run and contains test-like checks using the TF‑IDF fallback.

Testing strategy
- Unit tests use the TF‑IDF fallback and monkeypatch network calls to produce deterministic test results. Endpoint tests use TestClient with stubbed pipeline internals where appropriate.

Notes & next steps
- The current implementation is a strong demo and testable baseline. For production, persist FAISS indexes + metadata, add a robust file lock scheme, and move secrets to a secrets manager.

File references
- `src/rag_pipeline.py`, `src/server_api.py`, `src/config.py`, `src/cli.py`, `notebooks/multi_page_rag_notebook.ipynb`, `tests/`
