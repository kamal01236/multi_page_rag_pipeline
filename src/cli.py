"""
CLI entrypoint for the RAG pipeline using Ollama backend.
Example usage:

python -m src.cli \
  --urls https://en.wikipedia.org/wiki/Quantum_computing \
         https://en.wikipedia.org/wiki/Quantum_machine_learning \
  --ollama-url http://localhost:11434 \
  --ollama-token my-secret-token
"""

import argparse
import logging
from langchain.docstore.document import Document
from .rag_pipeline import (
    fetch_html,
    clean_wikipedia_html,
    randomized_chunks,
    build_vectorstore,
    build_retrieval_qa,
    make_page_specific_askers,
    detect_backend,
)
from .config import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_from_urls(urls, seed=None):
    docs = []
    for url in urls:
        logger.info("Fetching %s", url)
        html = fetch_html(url)
        title, cleaned = clean_wikipedia_html(html)
        chunks = randomized_chunks(cleaned, seed=seed)
        for i, c in enumerate(chunks):
            meta = {'source': url, 'title': title, 'chunk_index': i}
            docs.append(Document(page_content=c, metadata=meta))
        logger.info("Created %d chunks for %s", len(chunks), title)
    # Use CONFIG defaults when no explicit backend requested by caller.
    # Prefer emb_backend from CONFIG if set, else CONFIG.default_backend, else auto.
    from .config import CONFIG as _CONFIG
    chosen = _CONFIG.emb_backend or _CONFIG.default_backend or "auto"
    store, stype = build_vectorstore(docs, backend=chosen)
    logger.info("Built vector store: %s", stype)
    logger.info("Detected backend (helper): %s", detect_backend())
    return docs, store


def demo_pipeline(urls=None, seed=42, ollama_url=None, ollama_token=None, backend: str = "auto"):
    if urls is None:
        urls = [
            "https://en.wikipedia.org/wiki/Quantum_computing",
            "https://en.wikipedia.org/wiki/Quantum_machine_learning",
        ]
    docs, store = build_from_urls(urls, seed=seed)
    # Allow consumers to request a specific backend (openai/ollama/huggingface/tfidf/auto)
    qa, qatype, llm = build_retrieval_qa(store, ollama_url=ollama_url, ollama_token=ollama_token, backend=backend)
    # Pass the triple so askers can access the underlying llm for page-specific synthesis
    askers = make_page_specific_askers(store, (qa, qatype, llm), urls)
    return {
        "docs": docs,
        "store": store,
        "qa": qa,
        "qa_type": qatype,
        "llm": llm,
        "askers": askers,
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-page RAG pipeline demo (with Ollama)')
    parser.add_argument('--urls', nargs='+', required=True, help='Two or more URLs to ingest')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--ollama-token', type=str, default=None, help='Ollama API token (if required)')
    parser.add_argument('--emb-backend', type=str, default='auto', choices=['auto','openai','ollama','huggingface','tfidf'], help='Select embedding backend (vector store)')
    parser.add_argument('--llm-backend', type=str, default='auto', choices=['auto','openai','ollama','huggingface','tfidf'], help='Select LLM backend for RetrievalQA')
    parser.add_argument('--show-config', action='store_true', help='Pretty-print resolved configuration and exit')
    args = parser.parse_args()

    # Show resolved configuration at startup to help diagnose backend selection
    try:
        if args.show_config:
            try:
                import yaml
                print(yaml.safe_dump(CONFIG.__dict__, sort_keys=False))
            except Exception:
                print(CONFIG)
            return
        logger.info("Resolved CONFIG: %s", CONFIG)
    except Exception:
        # best-effort, don't fail the CLI if config printing fails
        logger.info("Resolved CONFIG: <unavailable>")

    docs, store = build_from_urls(args.urls, seed=args.seed)
    # Resolve effective backends: precedence CLI flag -> config value -> 'auto'
    emb_backend = args.emb_backend if args.emb_backend != 'auto' else (CONFIG.emb_backend or CONFIG.default_backend or 'auto')
    llm_backend = args.llm_backend if args.llm_backend != 'auto' else (CONFIG.llm_backend or CONFIG.default_backend or 'auto')

    if emb_backend and emb_backend != 'auto':
        store, stype = build_vectorstore(docs, backend=emb_backend)
        logger.info("Rebuilt vector store with emb-backend=%s: %s", emb_backend, stype)

    qa, qatype, llm = build_retrieval_qa(store, ollama_url=args.ollama_url, ollama_token=args.ollama_token, backend=llm_backend)

    print("QA type:", qatype)

    # demo queries
    queries = [
        "What are Quantum neural networks?",
        "What is the basic unit of information in quantum computing?",
    ]
    for q in queries:
        print("\nQUERY:", q)
        # Use the QA chain's invoke API where available.
        # Temporarily raise logger levels to WARNING to avoid noisy INFO logs
        root_logger = logging.getLogger()
        httpx_logger = logging.getLogger("httpx")
        openai_logger = logging.getLogger("openai")
        rag_logger = logging.getLogger("src.rag_pipeline")
        faiss_logger = logging.getLogger("faiss.loader")
        prev_levels = {
            "root": root_logger.level,
            "httpx": httpx_logger.level,
            "openai": openai_logger.level,
            "rag": rag_logger.level,
            "faiss": faiss_logger.level,
        }
        try:
            root_logger.setLevel(logging.WARNING)
            httpx_logger.setLevel(logging.WARNING)
            openai_logger.setLevel(logging.WARNING)
            rag_logger.setLevel(logging.WARNING)
            faiss_logger.setLevel(logging.WARNING)

            try:
                res = qa.invoke({"query": q})
            except Exception:
                # Some QA callables are simple functions
                res = qa(q)
        finally:
            # restore previous levels
            root_logger.setLevel(prev_levels["root"])
            httpx_logger.setLevel(prev_levels["httpx"])
            openai_logger.setLevel(prev_levels["openai"])
            rag_logger.setLevel(prev_levels["rag"])
            faiss_logger.setLevel(prev_levels["faiss"])

        # Print the result
        if isinstance(res, dict):
            print("ANSWER:\n", res.get("result") or res.get("answer") or res.get("output_text") or str(res))
            srcs = [d.metadata.get("source", "") for d in res.get("source_documents", [])] if res.get("source_documents") else res.get("sources", [])
            print("SOURCES:", srcs)
        else:
            print("ANSWER:\n", str(res))


if __name__ == '__main__':
    main()
