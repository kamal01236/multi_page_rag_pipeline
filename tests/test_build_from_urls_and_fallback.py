import types
import pytest

from src import rag_pipeline as rp


def stub_html(url):
    return """
    <html><h1 id='firstHeading'>Stub Page</h1>
    <div id='mw-content-text'><p>Sample content about qubits and quantum neural networks.</p></div>
    </html>
    """


@pytest.fixture(autouse=True)
def patch_fetch(monkeypatch):
    monkeypatch.setattr(rp, "fetch_html", lambda url: stub_html(url))
    yield


def test_build_from_urls_without_langchain(monkeypatch):
    # Simulate LangChain not available and ensure docs are created
    orig_doc = getattr(rp, "Document", None)
    orig_has_lang = getattr(rp, "HAS_LANGCHAIN", True)
    rp.Document = None
    rp.HAS_LANGCHAIN = False
    try:
        docs, store = rp.build_from_urls(["https://example.org/a"], seed=1)
        assert isinstance(docs, list) and len(docs) > 0
        d = docs[0]
        assert hasattr(d, "page_content") and hasattr(d, "metadata")
    finally:
        rp.Document = orig_doc
        rp.HAS_LANGCHAIN = orig_has_lang


def test_build_vectorstore_returns_tfidf_when_no_langchain(monkeypatch):
    orig_has_lang = getattr(rp, "HAS_LANGCHAIN", True)
    rp.HAS_LANGCHAIN = False
    try:
        docs = [types.SimpleNamespace(page_content="one two three", metadata={"source": "u"})]
        store, stype = rp.build_vectorstore(docs, backend="auto")
        assert stype == "tfidf"
        assert hasattr(store, "search")
        hits = store.search("one", k=1)
        assert len(hits) == 1
    finally:
        rp.HAS_LANGCHAIN = orig_has_lang
