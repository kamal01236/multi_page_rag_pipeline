"""
CLI entrypoint for the project. Example usage:

python -m src.cli --urls https://en.wikipedia.org/wiki/Quantum_computing https://en.wikipedia.org/wiki/Quantum_machine_learning
"""

import argparse
import logging
from .rag_pipeline import fetch_html, clean_wikipedia_html, randomized_chunks, build_vectorstore, build_retrieval_qa, make_page_specific_askers
from .rag_pipeline import detect_backend
from langchain.docstore.document import Document

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


def demo_pipeline(urls=None, seed=42):
    if urls is None:
        urls = [
            "https://en.wikipedia.org/wiki/Quantum_computing",
            "https://en.wikipedia.org/wiki/Quantum_machine_learning",
        ]
    docs, store = build_from_urls(urls, seed=seed)
    qa, qatype = build_retrieval_qa(store)
    askers = make_page_specific_askers(store, qa, urls)
    return {
        "docs": docs,
        "store": store,
        "qa": qa,
        "qa_type": qatype,
        "askers": askers,
    }

def main():
    parser = argparse.ArgumentParser(description='Multi-page RAG pipeline demo')
    parser.add_argument('--urls', nargs='+', required=True, help='Two or more URLs to ingest')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    docs, store = build_from_urls(args.urls, seed=args.seed)
    qa, qatype = build_retrieval_qa(store)
    print("QA type:", qatype)
    # demo queries
    queries = [
        "what are Quantum neural networks?",
        "What is the basic unit of information in quantum computing?",
    ]
    for q in queries:
        print("\\nQUERY:", q)
        if qatype == 'langchain':
            # run the langchain qa
            res = qa.invoke({"query": q})
            print("ANSWER:", res["result"])
            print("SOURCES:", [d.metadata["source"] for d in res["source_documents"]])
        else:
            res = qa(q, k=3)
            print("ANSWER:\\n", res['answer'][:1000])
            print("SOURCES:", res['sources'])

if __name__ == '__main__':
    main()
