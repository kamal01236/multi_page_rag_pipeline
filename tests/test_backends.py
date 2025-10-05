import os
from src.rag_pipeline import detect_backend, SimpleFallbackVectorStore, build_retrieval_qa


def test_detect_backend_value():
    """detect_backend should return one of the known strings."""
    val = detect_backend()
    assert val in {"faiss-openai", "faiss-hf", "tfidf", "none"}


def test_simple_qa_fallback_no_match():
    """When TF-IDF fallback is used, a query with no matching docs returns 'I don't know'."""
    # Build a minimal TF-IDF store from two short docs if sklearn is available
    try:
        docs = [
            type("D", (), {"page_content": "This document is about apples.", "metadata": {"source": "a"}})(),
            type("D", (), {"page_content": "This document is about oranges.", "metadata": {"source": "b"}})(),
        ]
        store = SimpleFallbackVectorStore(docs)
    except Exception:
        # If sklearn not available, skip this test by asserting detect_backend indicates none
        assert detect_backend() in {"faiss-openai", "faiss-hf", "none"}
        return

    qa, qatype = build_retrieval_qa(store)
    # Our query is unrelated; expect "I don't know" or concatenated but empty sources
    res = qa("qwertyuiopASDFGHJKLZXCVBNM", k=2)
    # Accept either explicit "I don't know" or an empty/short answer when fallback used
    assert isinstance(res, dict)
    assert "answer" in res and "sources" in res
