"""
CLI entrypoint for the Multi-Page RAG Pipeline.
Supports OpenAI, Ollama, HuggingFace, or TF-IDF backends.

Usage examples:
-----------------
python -m src.cli \
  --urls https://en.wikipedia.org/wiki/Quantum_computing \
         https://en.wikipedia.org/wiki/Quantum_machine_learning \
  --backend auto

You can later query interactively or extend to REST API.
"""

import argparse
import logging
from langchain.docstore.document import Document
from .rag_pipeline import (
    fetch_html,
    clean_wikipedia_html,
    chunk_text,
    build_vectorstore,
    make_page_specific_askers,
    search_db,
)
from .config import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_from_urls(urls, seed=None, backend="auto"):
    """Fetch, clean, chunk, and embed multiple URLs."""
    docs = []
    for url in urls:
        logger.info(f"Fetching {url}")
        html = fetch_html(url)
        text = clean_wikipedia_html(html)
        chunks = chunk_text(text, seed=seed)
        for i, c in enumerate(chunks):
            meta = {"source": url, "chunk_index": i}
            docs.append(Document(page_content=c, metadata=meta))
        logger.info("Created %d chunks for %s", len(chunks), url.split("/")[-1])

    store, stype = build_vectorstore(docs, backend=backend)
    logger.info("Built vector store: %s", stype)
    return docs, store, stype


def main():
    parser = argparse.ArgumentParser(description="Multi-page RAG pipeline CLI")
    parser.add_argument("--urls", nargs="+", required=True, help="List of URLs to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for chunking")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "openai", "ollama", "huggingface", "tfidf"],
        help="Select embedding backend",
    )
    parser.add_argument("--query", type=str, help="Ask a single query after ingestion")
    parser.add_argument("--page", type=str, help="Restrict query to specific page slug (optional)")
    args = parser.parse_args()

    # Build vector DB
    docs, store, backend_used = build_from_urls(args.urls, seed=args.seed, backend=args.backend)
    print(f"\n✅ Indexed {len(docs)} chunks using backend: {backend_used}\n")

    # Create per-page askers
    askers = make_page_specific_askers(store, args.urls)
    print(f"🧠 Created {len(askers)} page-specific askers\n")

    # Handle query if provided
    if args.query:
        print(f"🔎 Query: {args.query}\n")
        if args.page and args.page in askers:
            res = askers[args.page](args.query, k=3)
        else:
            hits = search_db(store, args.query, k=3)
            res = {
                "answer": "\n\n".join([h["chunk"] for h in hits]),
                "sources": list({h["source"] for h in hits if h.get("source")}),
            }
        print("🧾 Answer:\n", res["answer"])
        print("\n📚 Sources:", res["sources"])
    else:
        print("✅ Ingestion completed. Use --query to ask a question.")


if __name__ == "__main__":
    main()
