from src.rag_pipeline import SimpleFallbackVectorStore


def test_tfidf_ranking():
    docs = [
        type("D", (), {"page_content": "apple banana orange", "metadata": {"source": "a"}})(),
        type("D", (), {"page_content": "quantum computing qubit superposition", "metadata": {"source": "b"}})(),
    ]
    store = SimpleFallbackVectorStore(docs)
    hits = store.search("qubit", k=1)
    assert len(hits) == 1
    doc, score = hits[0]
    assert "qubit" in doc.page_content
