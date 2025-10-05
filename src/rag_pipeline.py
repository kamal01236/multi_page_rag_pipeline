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
    from langchain_ollama import OllamaLLM, OllamaEmbeddings  # ✅ new package
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


# ------------------ Fetch & Clean ------------------
def fetch_html(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": "multi-page-rag/1.0"})
    resp.raise_for_status()
    return resp.text


def clean_wikipedia_html(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1", id="firstHeading")
    title_text = title.get_text(strip=True) if title else ""
    content = soup.find("div", id="mw-content-text") or soup.find("article") or soup
    for tag in content.find_all(["table", "style", "script", "sup", "img", "aside"]):
        tag.decompose()

    stop_headings = {"References", "External links", "See also", "Further reading"}
    paragraphs = []
    for elem in content.find_all(["h2", "h3", "p", "li"]):
        if elem.name in ("h2", "h3"):
            heading = re.sub(r"\[.*?\]", "", elem.get_text(" ", strip=True))
            if any(h in heading for h in stop_headings):
                break
            continue
        text = elem.get_text(" ", strip=True)
        if not text:
            continue
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 30:
            paragraphs.append(text)
    return title_text, "\n\n".join(paragraphs)


# ------------------ Chunking ------------------
def randomized_chunks(text: str, min_size=MIN_CHUNK, max_size=MAX_CHUNK, overlap=OVERLAP, seed=None):
    if seed is not None:
        random.seed(seed)
    chunks, pos, n = [], 0, len(text)
    while pos < n:
        size = random.randint(min_size, max_size)
        chunk = text[pos:pos + size].strip()
        if not chunk:
            break
        chunks.append(chunk)
        pos += max(1, size - overlap)
        if len(chunks) > 10000:
            break
    return chunks


# ------------------ Vector Store ------------------
def build_vectorstore(docs: List[Any]):
    """
    Try building vectorstore in this order:
    1. Ollama embeddings (if FORCE_OLLAMA=1)
    2. OpenAI embeddings
    3. HuggingFace embeddings
    4. TF-IDF fallback
    """
    if not HAS_LANGCHAIN:
        raise RuntimeError("LangChain not installed.")

    force_ollama = os.getenv("FORCE_OLLAMA", "0").strip() == "1"
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_token = os.getenv("OLLAMA_TOKEN")

    # --- Try Ollama embeddings first if forced ---
    if force_ollama:
        try:
            logger.info("🚀 FORCE_OLLAMA=1 — using Ollama embeddings (mistral:instruct).")
            headers = {"Authorization": f"Bearer {ollama_token}"} if ollama_token else None
            emb = OllamaEmbeddings(base_url=ollama_url, model="mistral:instruct", headers=headers)
            store = FAISS.from_documents(docs, emb)
            return store, "faiss-ollama"
        except Exception as e:
            logger.warning(f"⚠️ Ollama embeddings failed: {e}")

    # --- Try OpenAI embeddings ---
    try:
        if os.getenv("OPENAI_API_KEY") and not force_ollama:
            logger.info("✅ Using OpenAI embeddings via FAISS.")
            emb = OpenAIEmbeddings(model="text-embedding-3-small")
            store = FAISS.from_documents(docs, emb)
            return store, "faiss-openai"
    except Exception as e:
        logger.warning(f"⚠️ OpenAI embeddings failed: {e}")

    # --- HuggingFace fallback ---
    try:
        logger.info("⚙️ Using HuggingFace sentence-transformers fallback.")
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        store = FAISS.from_documents(docs, emb)
        return store, "faiss-hf"
    except Exception as e:
        logger.warning(f"⚠️ HuggingFace embeddings failed: {e}")

    # --- TF-IDF fallback ---
    if HAS_SKLEARN:
        logger.warning("⚠️ Falling back to TF-IDF vector store.")

        class SimpleFallbackVectorStore:
            def __init__(self, docs):
                self.docs = docs
                texts = [d.page_content for d in docs]
                self.tfidf = TfidfVectorizer().fit(texts + ["placeholder"])
                self.vectors = self.tfidf.transform(texts).toarray()

            def search(self, query, k=3):
                qv = self.tfidf.transform([query]).toarray()[0]
                sims = (self.vectors @ qv) / (
                    (np.linalg.norm(self.vectors, axis=1) * (np.linalg.norm(qv) + 1e-9)) + 1e-9
                )
                idxs = sims.argsort()[::-1][:k]
                return [(self.docs[i], float(sims[i])) for i in idxs]

        return SimpleFallbackVectorStore(docs), "tfidf"

    raise RuntimeError("No valid embedding backend found.")


# ------------------ Backend Detector ------------------
def detect_backend() -> str:
    if os.getenv("FORCE_OLLAMA", "0") == "1":
        return "ollama"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.ok:
            return "ollama"
    except Exception:
        pass
    if HAS_LANGCHAIN:
        return "huggingface"
    if HAS_SKLEARN:
        return "tfidf"
    return "none"


# ------------------ RetrievalQA ------------------
def build_retrieval_qa(vector_store, ollama_url="http://localhost:11434", ollama_token=None):
    """
    Build a RetrievalQA chain with intelligent backend fallback:
    - FORCE_OLLAMA → Ollama
    - Try OpenAI (if API key works)
    - If rate limit or error → switch to Ollama
    - Else → use HuggingFace
    """
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant answering based strictly on the provided context.\n"
            "If you cannot answer based on the text, reply with: 'I don't know based on the provided documents.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer (include source URLs):"
        ),
    )

    retriever = getattr(vector_store, "as_retriever", lambda **_: vector_store)(search_kwargs={"k": 4})
    llm, backend = None, None

    # --- Check if FORCE_OLLAMA ---
    force_ollama = os.getenv("FORCE_OLLAMA", "0").strip() == "1"
    ollama_token = os.getenv("OLLAMA_TOKEN")
    logger.info(f"FORCE_OLLAMA={force_ollama}")

    if force_ollama:
        logger.info("🚀 FORCE_OLLAMA=1 — forcing local Ollama backend.")
        try:
            headers = {"Authorization": f"Bearer {ollama_token}"} if ollama_token else None
            llm = OllamaLLM(base_url=ollama_url, model="mistral:instruct", headers=headers)
            backend = "ollama"
            logger.info("✅ Using Ollama (mistral:instruct).")
        except Exception as e:
            logger.error(f"❌ Failed to init Ollama: {e}")
            raise
    else:
        llm = None
        backend = None

        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            try:
                logger.info("🔷 Trying OpenAI backend...")
                test_llm = OpenAI(temperature=0)
                _ = test_llm.invoke("ping")  # quick sanity call
                llm = test_llm
                backend = "openai"
            except Exception as e:
                logger.warning(f"⚠️ OpenAI failed ({type(e).__name__}): {e}")
                llm = None

        # Fallback to Ollama if OpenAI not available
        if llm is None:
            try:
                logger.info("🟢 Trying local Ollama backend...")
                r = requests.get(f"{ollama_url}/api/tags", timeout=2)
                if r.ok:
                    headers = {"Authorization": f"Bearer {ollama_token}"} if ollama_token else None
                    llm = OllamaLLM(base_url=ollama_url, model="mistral:instruct", headers=headers)
                    backend = "ollama"
                    logger.info("✅ Using local Ollama (mistral:instruct)")
            except Exception as e:
                logger.warning(f"⚠️ Ollama not reachable: {e}")

        # Fallback to HuggingFace
        if llm is None:
            try:
                from langchain_community.llms import HuggingFaceHub
                token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if token:
                    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.2})
                    backend = "huggingface"
                    logger.info("🧠 Using HuggingFaceHub fallback backend.")
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace fallback failed: {e}")

    if llm is None:
        raise RuntimeError("❌ No valid LLM backend available (OpenAI, Ollama, or HF).")

    logger.info(f"✅ Final LLM backend: {backend}")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    return qa_chain, backend


# ------------------ Page-specific askers ------------------
def make_page_specific_askers(vector_store, qa_callable, urls: List[str]):
    askers = {}
    for u in urls:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", u.rstrip('/').split('/')[-1]).lower()

        def make_fn(source_url):
            return lambda query, k=4: qa_callable.invoke({"query": query})

        askers[slug] = make_fn(u)
    return askers
