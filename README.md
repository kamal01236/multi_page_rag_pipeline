# Multi-Page RAG Pipeline (Industry-standard project, updated)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using two Wikipedia pages.
It includes fetching & cleaning, randomized overlapping chunking, FAISS-backed vector store, similarity search with optional source filtering, and a RetrievalQA module that cites sources and avoids hallucinations.

Now updated to **support multiple embedding/LLM backends**:
- **OpenAI (default)**: Requires `OPENAI_API_KEY`. High quality embeddings & fluent answers.
- **HuggingFace (free)**: Uses `sentence-transformers` for embeddings, and `transformers` for local LLMs if available.
- **Fallback TF-IDF**: Pure scikit-learn, no keys needed, lowest quality but works offline.

## Structure
```
multi_page_rag_project/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ src/
│  ├─ __init__.py
│  ├─ rag_pipeline.py
│  ├─ cli.py
│  └─ utils.py
├─ notebooks/
│  └─ demo_instructions.md
└─ tests/
   └─ test_chunking.py
```

## Quickstart
1. Create a Python 3.9+ virtual environment and install dependencies:
   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Export an OpenAI key (optional). If you have an OpenAI API key and want to use OpenAI embeddings/LLM set the environment variable before running (PowerShell example):
   ```powershell
   $env:OPENAI_API_KEY = "sk-..."
   ```

3. Run the demo script (example). The CLI will log which backend is selected (OpenAI / HuggingFace / TF-IDF fallback):
   ```powershell
   python -m src.cli --urls https://en.wikipedia.org/wiki/Quantum_computing https://en.wikipedia.org/wiki/Quantum_machine_learning
   ```

Notes:
- If `OPENAI_API_KEY` is present and LangChain is installed, the pipeline will prefer OpenAI embeddings + FAISS and will log "Using OpenAI embeddings/LLM".
- If LangChain is installed but no OpenAI key is present, the pipeline will try a HuggingFace sentence-transformer model and log "Using HuggingFace embeddings/LLM".
- If embeddings/FAISS cannot be constructed or LangChain is not available, a TF-IDF fallback is used (no keys required) and the pipeline will log "Using TF-IDF fallback vector store".
- The notebook in `notebooks/multi_page_rag_notebook.ipynb` contains a runnable demo that follows the same logic.

Approach and rationale
----------------------
This repository implements a robust, multi-backend RAG pipeline so it can run in a wide variety of environments:

- Primary: OpenAI embeddings + FAISS (best quality; requires `OPENAI_API_KEY`).
- Optional local LLM: Ollama (fast, runs locally) — can be forced with `FORCE_OLLAMA=1` or autodetected if reachable.
- Secondary: HuggingFace sentence-transformers when LangChain is present but no OpenAI key is configured.
- Fallback: TF-IDF (scikit-learn) vector store used when embeddings/FAISS or LangChain are not available.

Why this fallback chain?
- Portability: developers should be able to run experiments locally without API keys or heavy models.
- Cost and reliability: OpenAI delivers high-quality results but requires keys and may hit rate limits; Ollama lets teams run instruction-tuned models locally and is a cheap/repeatable option.
- Predictability: TF-IDF provides a deterministic, offline fallback so the ingestion/search pieces can still be validated in CI or constrained environments.

How the pipeline chooses a backend (high level)
- If `FORCE_OLLAMA=1` the system will try Ollama embeddings/LLM first.
- Else, if `OPENAI_API_KEY` is set and OpenAI calls succeed, the pipeline prefers OpenAI.
- If OpenAI isn't available or fails, the pipeline will try a local Ollama instance (if reachable) and then HuggingFaceHub.
- If those all fail, the pipeline falls back to TF-IDF.

Logging & observability
- The CLI and pipeline log the backend decisions and warnings; watch console output for lines like:
   - "Using OpenAI embeddings via FAISS." (OpenAI chosen)
   - "Using Ollama (mistral:instruct)." (Ollama chosen)
   - "Falling back to TF-IDF vector store." (fallback)

Running with Ollama (example)
```powershell
# set environment to prefer Ollama
$env:FORCE_OLLAMA = '1'
$env:OLLAMA_URL = 'http://localhost:11434'
# optionally if your Ollama requires a token
$env:OLLAMA_TOKEN = 'my-token'
python -m src.cli --urls https://en.wikipedia.org/wiki/Quantum_computing
```

Developer testing tips (suggested test cases)
- Unit: `tests/test_chunking.py` ensures chunking behavior.
- Backend detection: set `FORCE_OLLAMA=1` or `OPENAI_API_KEY` in test env and assert `detect_backend()` reports expected backend.
- Retrieval fallback: run the pipeline with only scikit-learn available and assert the QA fallback returns deterministic concatenated chunks when asked a query.
- Integration: run the demo end-to-end locally with small pages and validate that `askers` return answers with `sources` metadata.

Run the tests (after installing pytest):
```powershell
pip install pytest
pytest -q
```

## Backends
- **OpenAI**: export `OPENAI_API_KEY` before running to enable.
- **HuggingFace**: no key required; works with free local models.
- **Fallback (TF-IDF)**: automatic if neither OpenAI nor HuggingFace available.

