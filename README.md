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

## Backends
- **OpenAI**: export `OPENAI_API_KEY` before running to enable.
- **HuggingFace**: no key required; works with free local models.
- **Fallback (TF-IDF)**: automatic if neither OpenAI nor HuggingFace available.

