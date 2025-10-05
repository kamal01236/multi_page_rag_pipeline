# ...existing code...
"""
rag_pipeline.py — Final Version
Supports OpenAI, Ollama, HuggingFace, or TF-IDF embeddings/LLM for RAG pipeline.
Includes durable persistence helpers (index + metadata) with simple file-locking.
"""

import os
import re
import random
import logging
import requests
import json
import time
import pickle
import tempfile
import shutil
from typing import List, Tuple, Any, Optional
from bs4 import BeautifulSoup

# ------------------ LangChain Imports (optional) ------------------
try:
    from langchain.docstore.document import Document  # type: ignore
    from langchain.chains import RetrievalQA  # type: ignore
    from langchain.prompts import PromptTemplate  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_openai import OpenAIEmbeddings, OpenAI  # type: ignore
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    from langchain_ollama import OllamaLLM, OllamaEmbeddings  # type: ignore
    HAS_LANGCHAIN = True
except Exception:
    Document = None  # type: ignore
    HAS_LANGCHAIN = False

# ------------------ Fallbacks ------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    import numpy as np  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OVERLAP = 50
MIN_CHUNK = 400
MAX_CHUNK = 600


# ------------------ TF-IDF Fallback ------------------
if HAS_SKLEARN:
    class SimpleFallbackVectorStore:
        def __init__(self, docs: List[Any]):
            self.docs = docs
            texts = [d.page_content if hasattr(d, "page_content") else d["page_content"] for d in docs]
            self.tfidf = TfidfVectorizer().fit(texts + ["placeholder"])
            self.vectors = self.tfidf.transform(texts).toarray()

        def search(self, query: str, k: int = 3):
            qv = self.tfidf.transform([query]).toarray()[0]
            sims = (self.vectors @ qv) / (
                (np.linalg.norm(self.vectors, axis=1) * (np.linalg.norm(qv) + 1e-9)) + 1e-9
            )
            idxs = sims.argsort()[::-1][:k]
            return [(self.docs[i], float(sims[i])) for i in idxs]

        # persistence helpers for fallback
        def save_local(self, dirpath: str):
            os.makedirs(dirpath, exist_ok=True)
            np.save(os.path.join(dirpath, "vectors.npy"), self.vectors)
            with open(os.path.join(dirpath, "docs.pkl"), "wb") as f:
                pickle.dump(self.docs, f)
            with open(os.path.join(dirpath, "tfidf.pkl"), "wb") as f:
                pickle.dump(self.tfidf, f)

        @classmethod
        def load_local(cls, dirpath: str):
            with open(os.path.join(dirpath, "docs.pkl"), "rb") as f:
                docs = pickle.load(f)
            with open(os.path.join(dirpath, "tfidf.pkl"), "rb") as f:
                tfidf = pickle.load(f)
            inst = cls(docs)
            inst.tfidf = tfidf
            inst.vectors = np.load(os.path.join(dirpath, "vectors.npy"))
            return inst
else:
    class SimpleFallbackVectorStore:
        def __init__(self, docs: List[Any]):
            raise RuntimeError("scikit-learn not available for SimpleFallbackVectorStore")


# ------------------ Fetch + Clean ------------------
def fetch_html(url: str) -> str:
    for i in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": "multi-page-rag/1.0"}, timeout=10)
            if r.status_code == 200:
                return r.text
            logger.warning("Fetch returned status %s for %s", r.status_code, url)
        except Exception as e:
            logger.warning("Fetch attempt %d failed for %s: %s", i + 1, url, e)
    raise RuntimeError(f"Failed to fetch {url}")


def clean_wikipedia_html(html: str) -> Tuple[str, str]:
    """
    Clean and extract main text + title from a Wikipedia-like page.
    Returns: (title, cleaned_text)
    """
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1", id="firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else ""
    content = soup.find("div", id="mw-content-text") or soup.find("article") or soup
    for tag in content.find_all(["table", "style", "script", "sup", "img", "figure", "span", "math"]):
        tag.decompose()
    stop_headings = {"References", "External links", "See also", "Further reading"}
    paragraphs = []
    for elem in content.find_all(["h2", "h3", "p", "li"]):
        if elem.name in ("h2", "h3"):
            heading = elem.get_text(" ", strip=True)
            if any(h in heading for h in stop_headings):
                break
            paragraphs.append(f"\n## {heading}\n")
            continue
        text = elem.get_text(" ", strip=True)
        if not text:
            continue
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 30:
            paragraphs.append(text)
    clean_text = "\n\n".join(paragraphs).strip()
    return title, clean_text


# ------------------ Chunking ------------------
def chunk_text(text: str, min_size: int = MIN_CHUNK, max_size: int = MAX_CHUNK, overlap: int = OVERLAP, seed: Optional[int] = None) -> List[str]:
    if seed is not None:
        random.seed(seed)
    chunks: List[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        size = random.randint(min_size, max_size)
        chunk = text[pos: pos + size].strip()
        if not chunk:
            break
        chunks.append(chunk)
        pos += max(1, size - overlap)
        if len(chunks) > 10000:
            break
    return chunks


# ------------------ Vector Store Builder ------------------
def build_vectorstore(docs: List[Any], backend: str = "auto"):
    """Build FAISS (via LangChain) or deterministic TF-IDF fallback."""
    # If langchain present, try to build FAISS using preferred backends
    if HAS_LANGCHAIN:
        def try_backend(name: str):
            try:
                if name == "openai" and os.getenv("OPENAI_API_KEY"):
                    logger.info("Using OpenAI embeddings.")
                    emb = OpenAIEmbeddings(model="text-embedding-3-small")
                    return FAISS.from_documents(docs, emb), "faiss-openai"
                if name == "ollama":
                    logger.info("Using Ollama embeddings.")
                    emb = OllamaEmbeddings(model="mistral:instruct")
                    return FAISS.from_documents(docs, emb), "faiss-ollama"
                if name == "huggingface":
                    logger.info("Using HuggingFace embeddings.")
                    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    return FAISS.from_documents(docs, emb), "faiss-hf"
            except Exception as e:
                logger.warning("%s backend failed: %s", name, e)
            return None

        # explicit backend precedence
        if backend and backend != "auto":
            res = try_backend(backend)
            if res:
                return res
            raise RuntimeError(f"Requested backend '{backend}' failed to initialize")

        # auto selection
        candidates = ["openai", "ollama", "huggingface"]
        for cand in candidates:
            res = try_backend(cand)
            if res:
                return res

    # Fallback to TF-IDF (deterministic)
    if HAS_SKLEARN:
        logger.warning("⚠️ Falling back to TF-IDF vector store.")
        return SimpleFallbackVectorStore(docs), "tfidf"

    raise RuntimeError("No vector backend available.")


# ------------------ Search ------------------
def search_db(vector_store, query: str, k: int = 3, filter_source: Optional[str] = None):
    """
    Search and post-filter results by source substring (case-insensitive).
    Returns list of dicts: { 'chunk', 'score', 'source', 'id' (optional) }
    """
    try:
        if hasattr(vector_store, "similarity_search_with_score"):
            docs_and_scores = vector_store.similarity_search_with_score(query, k * 3)
        elif hasattr(vector_store, "similarity_search"):
            docs = vector_store.similarity_search(query, k * 3)
            docs_and_scores = [(d, 1.0) for d in docs]
        elif hasattr(vector_store, "search"):
            docs_and_scores = vector_store.search(query, k * 3)
        else:
            raise RuntimeError("Unsupported vector store: missing search interface")

        filtered = []
        for doc, score in docs_and_scores:
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source")
            if filter_source and src and filter_source.lower() not in (src or "").lower():
                continue
            filtered.append({
                "chunk": getattr(doc, "page_content", str(doc)),
                "score": float(score),
                "source": src,
                "id": meta.get("id")
            })
            if len(filtered) >= k:
                break

        logger.info("search_db: q=%s k=%d returned=%d filter=%s", query, k, len(filtered), filter_source)
        return filtered

    except Exception as e:
        logger.exception("search_db() failed: %s", e)
        return []


# ------------------ build_from_urls ------------------
def build_from_urls(urls: List[str], seed: Optional[int] = None):
    """Fetch, clean, chunk, and index pages. Returns (docs, vector_store)."""
    random.seed(seed)
    all_docs: List[Any] = []

    # Lightweight fallback Document
    class _SimpleDoc:
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

    for url in urls:
        logger.info("Fetching %s", url)
        html = fetch_html(url)
        title, clean_text = clean_wikipedia_html(html)
        chunks = chunk_text(clean_text, seed=seed)
        for i, chunk in enumerate(chunks):
            meta = {"source": url, "chunk_index": i, "title": title}
            if Document is not None:
                all_docs.append(Document(page_content=chunk, metadata=meta))
            else:
                all_docs.append(_SimpleDoc(page_content=chunk, metadata=meta))
        logger.info("Created %d chunks for %s", len(chunks), url.split("/")[-1])

    store, stype = build_vectorstore(all_docs)
    logger.info("Built vector store: %s", stype)
    return all_docs, store


# ------------------ Page-specific Askers ------------------
def make_page_specific_askers(vector_store, urls: List[str]):
    """Generate per-page restricted askers. urls is canonical list of source URLs."""
    askers = {}

    def make_fn(source_url: str):
        def fn(query: str, k: int = 3):
            hits = search_db(vector_store, query, k=k, filter_source=source_url)
            if not hits:
                return {"answer": "I don't know based on the provided documents.", "sources": []}
            sources = list(dict.fromkeys([h["source"] for h in hits if h.get("source")]))
            parts = [f"SOURCE: {h['source']}\nCONTENT: {h['chunk']}" for h in hits]
            return {"answer": "\n\n".join(parts), "sources": sources}
        return fn

    for u in urls:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", u.split("/")[-1]).lower()
        askers[slug] = make_fn(u)

    return askers


# ------------------ Persistence helpers (index + metadata) ------------------
def _acquire_lock(path: str, timeout: float = 10.0):
    lockdir = f"{path}.lockdir"
    start = time.time()
    while True:
        try:
            os.mkdir(lockdir)
            return lockdir
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout acquiring lock for {path}")
            time.sleep(0.1)


def _release_lock(lockdir: str):
    try:
        os.rmdir(lockdir)
    except Exception:
        try:
            shutil.rmtree(lockdir)
        except Exception:
            pass


def persist_index(vector_store, index_dir: str, docs: Optional[List[Any]] = None, metadata_jsonl: Optional[str] = None, timeout: float = 10.0):
    """
    Persist the vector index and per-chunk metadata.
    - index_dir: directory to write vector index files
    - docs: list of doc objects (used to build metadata jsonl if provided)
    - metadata_jsonl: path to write metadata (atomic write)
    """
    os.makedirs(index_dir, exist_ok=True)
    lock = _acquire_lock(index_dir, timeout=timeout)
    try:
        # FAISS via LangChain-like store
        if hasattr(vector_store, "save_local"):
            try:
                vector_store.save_local(index_dir)
            except Exception:
                # some vectorstores expose different API, attempt attr-based save
                if hasattr(vector_store, "save"):
                    vector_store.save(index_dir)
                else:
                    logger.warning("Vector store has no standard save_local/save method.")
        elif isinstance(vector_store, SimpleFallbackVectorStore):
            try:
                vector_store.save_local(index_dir)
            except Exception as e:
                logger.warning("Failed to save TF-IDF store: %s", e)
        else:
            logger.warning("Unknown vector_store type; skipping index persistence.")

        # persist metadata atomically
        if docs and metadata_jsonl:
            tmp = f"{metadata_jsonl}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for d in docs:
                    meta = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})
                    out = {
                        "source": meta.get("source"),
                        "title": meta.get("title"),
                        "chunk_index": meta.get("chunk_index"),
                        "ingest_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "id": meta.get("id"),
                    }
                    fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            os.replace(tmp, metadata_jsonl)
    finally:
        _release_lock(lock)


def load_persisted_index(index_dir: str, metadata_jsonl: Optional[str] = None):
    """
    Load a persisted index + metadata (best-effort).
    Returns (vector_store, metadata_list)
    """
    if not os.path.exists(index_dir):
        raise FileNotFoundError(index_dir)
    # Try FAISS/langchain style
    try:
        if HAS_LANGCHAIN and 'FAISS' in globals() and hasattr(FAISS, "load_local"):
            # langchain_community FAISS has load_local classmethod
            store = FAISS.load_local(index_dir)
            meta = []
            if metadata_jsonl and os.path.exists(metadata_jsonl):
                with open(metadata_jsonl, "r", encoding="utf-8") as fh:
                    meta = [json.loads(l) for l in fh]
            return store, meta
    except Exception:
        pass

    # Try TF-IDF fallback files
    if HAS_SKLEARN:
        try:
            store = SimpleFallbackVectorStore.load_local(index_dir)
            meta = []
            if metadata_jsonl and os.path.exists(metadata_jsonl):
                with open(metadata_jsonl, "r", encoding="utf-8") as fh:
                    meta = [json.loads(l) for l in fh]
            return store, meta
        except Exception:
            pass

    raise RuntimeError("Could not load persisted index from " + index_dir)


# ------------------ Perfect Retrieval QA Builder ------------------
def build_retrieval_qa(vector_store, backend: str = "auto"):
    """
    Builds RetrievalQA: attempts LLM backends, else returns deterministic simple_qa.
    """
    logger = logging.getLogger("retrieval_qa")
    logger.info("Building Retrieval QA with backend=%s", backend)

    # Prompt template for LLMs (LangChain)
    template = (
        "You are a highly factual assistant. "
        "Use ONLY the provided context to answer the question. "
        "If you don't know, say: 'I don't know based on the provided documents.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer (include relevant source URLs if present):"
    )

    # Try LLM backends (OpenAI, Ollama, HuggingFace)
    llm = None
    backend_used = "unknown"
    tried = []

    def try_openai():
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        from langchain_openai import OpenAI  # type: ignore
        return OpenAI(temperature=0), "openai"

    def try_ollama():
        from langchain_ollama import OllamaLLM  # type: ignore
        return OllamaLLM(model="mistral:instruct"), "ollama"

    def try_huggingface():
        from langchain_community.llms import HuggingFaceHub  # type: ignore
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set")
        return HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0}), "huggingface"

    for name, fn in [("openai", try_openai), ("ollama", try_ollama), ("huggingface", try_huggingface)]:
        if backend not in ("auto", name):
            continue
        try:
            llm, backend_used = fn()
            break
        except Exception as e:
            tried.append(f"{name}→{e}")
            continue

    # No LLM -> simple deterministic QA
    if llm is None:
        logger.warning("No LLM backend available. Using TF-IDF/simple fallback. Tried: %s", tried)

        def simple_qa(query: str, k: int = 3, filter_source: Optional[str] = None):
            hits = search_db(vector_store, query, k=k, filter_source=filter_source)
            if not hits:
                return {"answer": "I don't know based on the provided documents.", "sources": [], "confidence": 0.0}
            answer = "\n\n".join([h["chunk"] for h in hits])
            sources = list({h["source"] for h in hits if h.get("source")})
            conf = float(sum(h["score"] for h in hits) / max(1, len(hits)))
            return {"answer": answer, "sources": sources, "confidence": conf}

        return simple_qa, "tfidf", None

    # Build retriever
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    except Exception:
        retriever = getattr(vector_store, "search", vector_store)

    # Create RetrievalQA chain (LangChain)
    try:
        from langchain.prompts import PromptTemplate  # type: ignore
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    except Exception:
        prompt = None

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt} if prompt is not None else {},
        )
    except Exception as e:
        logger.warning("Failed to instantiate RetrievalQA chain: %s", e)
        return build_retrieval_qa(vector_store, backend="tfidf")[0], "tfidf", None

    def qa_callable(query: str, k: int = 3, filter_source: Optional[str] = None):
        try:
            # If filter_source provided, perform a pre-search and pass concatenated context to LLM
            if filter_source:
                hits = search_db(vector_store, query, k=k, filter_source=filter_source)
                if not hits:
                    return {"answer": "I don't know based on the provided documents.", "sources": [], "confidence": 0.0}
                context = "\n\n".join([f"Source: {h['source']}\n\n{h['chunk']}" for h in hits])
                prompt_input = {"query": query, "context": context}
                res = qa_chain.invoke({"query": query, "context": context})
            else:
                # Let retrieval happen inside chain
                res = qa_chain.invoke({"query": query})

            answer = res.get("result") or res.get("output_text") or str(res)
            src_docs = res.get("source_documents", []) or []
            sources = sorted({getattr(d, "metadata", {}).get("source", None) for d in src_docs if getattr(d, "metadata", None)})
            return {"answer": answer.strip(), "sources": [s for s in sources if s], "confidence": 1.0 if answer else 0.0}
        except Exception as e:
            logger.warning("QA chain failed: %s", e)
            # fallback to simple retrieval
            return build_retrieval_qa(vector_store, backend="tfidf")[0](query, k, filter_source)

    return qa_callable, backend_used, llm
# ...existing code...