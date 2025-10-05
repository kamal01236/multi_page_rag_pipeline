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
    store, stype = build_vectorstore(docs)
    logger.info("Built vector store: %s", stype)
    logger.info("Detected backend (helper): %s", detect_backend())
    return docs, store


def demo_pipeline(urls=None, seed=42, ollama_url=None, ollama_token=None):
    if urls is None:
        urls = [
            "https://en.wikipedia.org/wiki/Quantum_computing",
            "https://en.wikipedia.org/wiki/Quantum_machine_learning",
        ]
    docs, store = build_from_urls(urls, seed=seed)
    qa, qatype, llm = build_retrieval_qa(store, ollama_url=ollama_url, ollama_token=ollama_token)
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
    args = parser.parse_args()

    docs, store = build_from_urls(args.urls, seed=args.seed)
    qa, qatype, llm = build_retrieval_qa(store, ollama_url=args.ollama_url, ollama_token=args.ollama_token)

    print("QA type:", qatype)

    # demo queries
    queries = [
        "What are Quantum neural networks?",
        "What is the basic unit of information in quantum computing?",
    ]
    for q in queries:
        print("\nQUERY:", q)
        # Use the QA chain's invoke API where available
        try:
            res = qa.invoke({"query": q})
            if isinstance(res, dict):
                print("ANSWER:\n", res.get("result") or res.get("output_text") or str(res))
                srcs = [d.metadata.get("source", "") for d in res.get("source_documents", [])] if res.get("source_documents") else []
                print("SOURCES:", srcs)
            else:
                print("ANSWER:\n", str(res))
        except Exception as e:
            print("QA invocation failed:", e)


if __name__ == '__main__':
    main()
