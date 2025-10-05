"""
rag_pipeline.py — Final Version
Supports OpenAI, Ollama, HuggingFace, or TF-IDF embeddings/LLM for RAG pipeline.
"""

import os, re, random, logging, requests
from typing import List, Tuple, Any
from bs4 import BeautifulSoup

# ------------------ LangChain Imports ------------------
try:
    from langchain.docstore.document import Document
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, OpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False

# ------------------ Fallbacks ------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
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
        def __init__(self, docs):
            self.docs = docs
            texts = [d.page_content if hasattr(d, "page_content") else d["page_content"] for d in docs]
            self.tfidf = TfidfVectorizer().fit(texts + ["placeholder"])
            self.vectors = self.tfidf.transform(texts).toarray()

        def search(self, query, k=3):
            qv = self.tfidf.transform([query]).toarray()[0]
            sims = (self.vectors @ qv) / (
                (np.linalg.norm(self.vectors, axis=1) * (np.linalg.norm(qv) + 1e-9)) + 1e-9
            )
            idxs = sims.argsort()[::-1][:k]
            return [(self.docs[i], float(sims[i])) for i in idxs]
else:
    class SimpleFallbackVectorStore:
        def __init__(self, docs):
            raise RuntimeError("scikit-learn not available for SimpleFallbackVectorStore")


# ------------------ Fetch + Clean ------------------
def fetch_html(url: str) -> str:
    for i in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": "multi-page-rag/1.0"}, timeout=10)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            logger.warning(f"Fetch attempt {i+1} failed for {url}: {e}")
    raise RuntimeError(f"Failed to fetch {url}")


def clean_wikipedia_html(html: str) -> Tuple[str, str]:
    """
    Clean and extract main text + title from a Wikipedia-like page.

    - Removes tables, figures, scripts, references, infoboxes, etc.
    - Keeps section flow (h2/h3 headings as markers).
    - Removes bracketed citations like [1], [citation needed].
    - Stops at 'References', 'External links', or 'See also' sections.
    Returns: (title, cleaned_text)
    """
    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title_tag = soup.find("h1", id="firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Identify content region
    content = soup.find("div", id="mw-content-text") or soup.find("article") or soup

    # Remove unwanted elements
    for tag in content.find_all(["table", "style", "script", "sup", "img", "figure", "span", "math"]):
        tag.decompose()

    # Define stop headings
    stop_headings = {"References", "External links", "See also", "Further reading"}

    paragraphs = []
    for elem in content.find_all(["h2", "h3", "p", "li"]):
        # Stop parsing after reference sections
        if elem.name in ("h2", "h3"):
            heading = elem.get_text(" ", strip=True)
            if any(h in heading for h in stop_headings):
                break
            paragraphs.append(f"\n## {heading}\n")
            continue

        # Extract paragraph or list text
        text = elem.get_text(" ", strip=True)
        if not text:
            continue

        # Remove reference marks like [1], [citation needed], etc.
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 30:
            paragraphs.append(text)

    clean_text = "\n\n".join(paragraphs).strip()
    return title, clean_text


# ------------------ Chunking ------------------
def chunk_text(text: str, min_size=MIN_CHUNK, max_size=MAX_CHUNK, overlap=OVERLAP, seed=None):
    if seed is not None:
        random.seed(seed)
    chunks = []
    pos = 0
    while pos < len(text):
        size = random.randint(min_size, max_size)
        chunk = text[pos:pos + size]
        chunks.append(chunk)
        pos += size - overlap
    return chunks


# ------------------ Vector Store Builder ------------------
def build_vectorstore(docs: List[Any], backend: str = "auto"):
    """Build FAISS or fallback vector store."""
    if not HAS_LANGCHAIN:
        raise RuntimeError("LangChain not installed.")

    def try_backend(name):
        try:
            if name == "openai" and os.getenv("OPENAI_API_KEY"):
                logger.info("Using OpenAI embeddings.")
                emb = OpenAIEmbeddings(model="text-embedding-3-small")
                return FAISS.from_documents(docs, emb), "faiss-openai"
            elif name == "ollama":
                logger.info("Using Ollama embeddings.")
                emb = OllamaEmbeddings(model="mistral:instruct")
                return FAISS.from_documents(docs, emb), "faiss-ollama"
            elif name == "huggingface":
                logger.info("Using HuggingFace embeddings.")
                emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                return FAISS.from_documents(docs, emb), "faiss-hf"
        except Exception as e:
            logger.warning(f"{name} backend failed: {e}")
            return None

    # Try ordered backends
    for candidate in ["openai", "ollama", "huggingface"]:
        res = try_backend(candidate)
        if res:
            return res

    # Fallback to TF-IDF
    if HAS_SKLEARN:
        logger.warning("⚠️ Falling back to TF-IDF vector store.")
        return SimpleFallbackVectorStore(docs), "tfidf"

    raise RuntimeError("No vector backend available.")


# ------------------ Search ------------------
def search_db(vector_store, query: str, k: int = 3, filter_source: str = None):
    """
    Perform similarity search across supported vector store backends (FAISS, Ollama, OpenAI, HuggingFace, TF-IDF).
    Supports both LangChain-style and fallback vector stores.
    """
    try:
        # ✅ LangChain FAISS or other vectorstore with .similarity_search_with_score
        if hasattr(vector_store, "similarity_search_with_score"):
            docs_and_scores = vector_store.similarity_search_with_score(query, k * 3)
        # ✅ LangChain FAISS older style
        elif hasattr(vector_store, "similarity_search"):
            docs = vector_store.similarity_search(query, k * 3)
            docs_and_scores = [(d, 1.0) for d in docs]
        # ✅ TF-IDF fallback
        elif hasattr(vector_store, "search"):
            docs_and_scores = vector_store.search(query, k * 3)
        else:
            raise RuntimeError("Unsupported vector store: missing similarity search interface")

        filtered = []
        for doc, score in docs_and_scores:
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source")
            if filter_source and src and filter_source not in src:
                continue
            filtered.append({
                "chunk": getattr(doc, "page_content", str(doc)),
                "score": float(score),
                "source": src,
            })
            if len(filtered) >= k:
                break
        return filtered

    except Exception as e:
        logger.exception(f"search_db() failed: {e}")
        return []


# ------------------ build_from_urls ------------------
def build_from_urls(urls, seed=None):
    """Fetch, clean, chunk, and index pages."""
    random.seed(seed)
    all_docs = []
    for url in urls:
        logger.info(f"Fetching {url}")
        html = fetch_html(url)
        title, clean_text = clean_wikipedia_html(html)
        chunks = chunk_text(clean_text)
        for i, chunk in enumerate(chunks):
            all_docs.append(Document(page_content=chunk, metadata={"source": url, "chunk_index": i}))
        logger.info(f"Created {len(chunks)} chunks for {url.split('/')[-1]}")

    store, stype = build_vectorstore(all_docs)
    logger.info(f"Built vector store: {stype}")
    return all_docs, store


# ------------------ Page-specific Askers ------------------
def make_page_specific_askers(vector_store, urls: List[str]):
    """Generate per-page restricted askers."""
    askers = {}

    def make_fn(source_url):
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

# ------------------ Perfect Retrieval QA Builder ------------------
def build_retrieval_qa(vector_store, backend: str = "auto"):
    """
    Builds an intelligent RetrievalQA system that:
    - Uses OpenAI, Ollama, or HuggingFace for synthesis
    - Falls back to TF-IDF simple retrieval if no LLM available
    - Always returns structured JSON with answer, sources, and confidence
    """

    import logging, time
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_community.vectorstores import FAISS

    logger = logging.getLogger("retrieval_qa")
    logger.info("🔧 Building Retrieval QA with backend=%s", backend)

    # --- Prompt template ---
    template = (
        "You are a highly factual assistant. "
        "Use ONLY the provided context to answer the question. "
        "If you don't know, say: 'I don't know based on the provided documents.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer (include relevant source URLs if present):"
    )
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # --- Backend selection helpers ---
    llm = None
    backend_used = "unknown"

    def try_openai():
        from langchain_openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        logger.info("✅ Using OpenAI backend")
        return OpenAI(temperature=0), "openai"

    def try_ollama():
        from langchain_ollama import OllamaLLM
        logger.info("✅ Using Ollama backend (model=mistral:instruct)")
        return OllamaLLM(model="mistral:instruct"), "ollama"

    def try_huggingface():
        from langchain_community.llms import HuggingFaceHub
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set")
        logger.info("✅ Using HuggingFace backend (tiiuae/falcon-7b-instruct)")
        return HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0}), "huggingface"

    # --- Try ordered backends ---
    tried = []
    for name, fn in [("openai", try_openai), ("ollama", try_ollama), ("huggingface", try_huggingface)]:
        if backend not in ("auto", name):
            continue
        try:
            llm, backend_used = fn()
            break
        except Exception as e:
            tried.append(f"{name}→{e}")
            continue

    # --- Fallback to simple retrieval ---
    if llm is None:
        logger.warning("⚠️ No LLM backend available. Using TF-IDF/simple fallback. Tried: %s", tried)

        def simple_qa(query: str, k: int = 3, filter_source: str = None):
            hits = search_db(vector_store, query, k=k, filter_source=filter_source)
            if not hits:
                return {
                    "answer": "I don't know based on the provided documents.",
                    "sources": [],
                    "confidence": 0.0,
                }
            answer = "\n\n".join([h["chunk"] for h in hits])
            sources = list({h["source"] for h in hits if h.get("source")})
            conf = sum(h["score"] for h in hits) / len(hits)
            return {"answer": answer, "sources": sources, "confidence": conf}

        return simple_qa, "tfidf", None

    # --- Build retriever from vector store ---
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    except Exception:
        retriever = getattr(vector_store, "search", vector_store)

    # --- Create RetrievalQA chain ---
    logger.info("🔗 Creating RetrievalQA chain using %s backend", backend_used)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    def qa_callable(query: str, k: int = 3, filter_source: str = None):
        """Unified callable for answering queries."""
        try:
            res = qa_chain.invoke({"query": query})
            answer = res.get("result") or res.get("output_text") or str(res)
            src_docs = res.get("source_documents", [])
            sources = sorted(
                {getattr(d, "metadata", {}).get("source", None) for d in src_docs if getattr(d, "metadata", None)}
            )
            return {
                "answer": answer.strip(),
                "sources": [s for s in sources if s],
                "confidence": 1.0 if answer else 0.0,
            }
        except Exception as e:
            logger.warning("⚠️ QA chain failed: %s", e)
            # fallback to direct search
            return build_retrieval_qa(vector_store, backend="tfidf")[0](query, k, filter_source)

    return qa_callable, backend_used, llm
