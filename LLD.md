# Low-Level Design (LLD) for Multi-Page RAG Pipeline

This document describes the components and flow for the multi-page RAG pipeline implemented in the repository.

Components

- fetch_html(url): simple HTTP fetch with a polite User-Agent.
- clean_wikipedia_html(html): uses BeautifulSoup to extract main page content, removes scripts, tables, superscripts, and stops at common reference sections.
- randomized_chunks(text): produces overlapping chunks; chunk size is randomized between MIN_CHUNK and MAX_CHUNK (400-600 chars by default) with a fixed overlap (50 chars).
- Document construction: each chunk becomes a LangChain Document with metadata {source, title, chunk_index}.
- build_vectorstore(docs): tries FAISS with OpenAI embeddings, falls back to HF embeddings, and finally to a TF-IDF fallback (scikit-learn) if embeddings are unavailable.
	- The system will log which backend is used. Preference order:
		1. OpenAI (requires OPENAI_API_KEY and LangChain available) — logs: "Using OpenAI embeddings/LLM".
		2. HuggingFace (LangChain + sentence-transformers) — logs: "Using HuggingFace embeddings/LLM".
		3. TF-IDF fallback (scikit-learn) — logs: "Using TF-IDF fallback vector store".
- search_db(vector_store, query, k, filter_source): returns top-k matching chunks and their source URL; supports optional post-retrieval source filtering.
- build_retrieval_qa(vector_store): builds a RetrievalQA chain when LangChain is available; creates a conservative prompt to discourage hallucinations and to require source listing. Returns a callable that always returns a dict {answer, sources}.
	- When building the QA wrapper the code will log which QA path is used (LangChain LLM wrapper vs simple concatenation fallback). The QA wrapper enforces:
		- temperature=0 where supported to reduce randomness.
		- A conservative prompt instructing: "Use ONLY the information in the provided context" and to reply "I don't know" if the information isn't present.
		- When running, the wrapper returns a standardized dict: {answer, sources} so callers can programmatically inspect provenance.
- make_page_specific_askers(vector_store, qa_callable, urls): factory that returns per-page askers that call the QA with a post-retrieval source filter.

Quality & Safety

- LLMs are invoked with temperature=0 where possible to reduce randomness.
- The prompt explicitly instructs the model to "Use ONLY the information in the provided context" and to respond with "I don't know" if information is missing.
- For environments without LangChain or embeddings, a TF-IDF fallback is used (best-effort).

Notes

- The implementation aims for portability: it will run entirely locally with TF-IDF if OpenAI keys or models are not present.
- The system preserves chunk-level provenance via metadata so answers can cite the exact source page(s).

