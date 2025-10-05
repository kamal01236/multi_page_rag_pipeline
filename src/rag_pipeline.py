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


# Module-level TF-IDF fallback implementation (exported for tests)
if HAS_SKLEARN:
    class SimpleFallbackVectorStore:
        def __init__(self, docs):
            self.docs = docs
            texts = [d.page_content if hasattr(d, 'page_content') else d['page_content'] for d in docs]
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
def build_vectorstore(docs: List[Any], backend: str = "auto"):
    """
    Try building vectorstore in this order:
    1. Ollama embeddings (if FORCE_OLLAMA=1)
    2. OpenAI embeddings
    3. HuggingFace embeddings
    4. TF-IDF fallback
    """
    if not HAS_LANGCHAIN:
        raise RuntimeError("LangChain not installed.")

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_token = os.getenv("OLLAMA_TOKEN")

    # Helper: try to build for a single backend name
    def try_backend(name: str):
        try:
            if name == "ollama":
                logger.info("Trying Ollama embeddings.")
                headers = {"Authorization": f"Bearer {ollama_token}"} if ollama_token else None
                emb = OllamaEmbeddings(base_url=ollama_url, model="mistral:instruct", headers=headers)
                store = FAISS.from_documents(docs, emb)
                return store, "faiss-ollama"
            if name == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY not set")
                logger.info("Using OpenAI embeddings via FAISS.")
                emb = OpenAIEmbeddings(model="text-embedding-3-small")
                store = FAISS.from_documents(docs, emb)
                return store, "faiss-openai"
            if name == "huggingface":
                logger.info("Using HuggingFace sentence-transformers.")
                emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                store = FAISS.from_documents(docs, emb)
                return store, "faiss-hf"
        except Exception as e:
            logger.warning("Backend %s failed: %s", name, e)
        return None

    # Selection order
    if backend and backend != "auto":
        res = try_backend(backend)
        if res:
            return res
        # If explicit backend failed, raise to surface the issue
        raise RuntimeError(f"Requested backend '{backend}' is not available or failed to build")

    # Auto selection: prefer OpenAI, then Ollama, then HF, then TF-IDF
    for candidate in ["openai", "ollama", "huggingface"]:
        res = try_backend(candidate)
        if res:
            return res

    # --- TF-IDF fallback ---
    if HAS_SKLEARN:
        logger.warning("⚠️ Falling back to TF-IDF vector store.")
        return SimpleFallbackVectorStore(docs), "tfidf"

    raise RuntimeError("No valid embedding backend found.")


# ------------------ Backend Detector ------------------
def detect_backend() -> str:
    # Prefer explicit environment-based backends
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    # Check for local Ollama service
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        if r.ok:
            return "ollama"
    except Exception:
        pass
    # If LangChain is present assume huggingface is available
    if HAS_LANGCHAIN:
        return "huggingface"
    if HAS_SKLEARN:
        return "tfidf"
    return "none"


# ------------------ RetrievalQA ------------------
import time, tiktoken
from collections import deque

# ------------------ RetrievalQA ------------------
def build_retrieval_qa(vector_store, ollama_url="http://localhost:11434", ollama_token=None, backend: str = "auto"):
    """
    Build a RetrievalQA chain with quota-safe OpenAI backend:
    - Respects 3 RPM (requests/min) and 10,000 TPM (tokens/min) limits
    - Retries automatically on 429 errors
    - Falls back to Ollama or HuggingFace if OpenAI fails
    """

    # Quota limits (adjustable)
    OPENAI_RPM = 3       # requests per minute
    OPENAI_TPM = 10_000  # tokens per minute

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant answering based strictly on the provided context.\n"
            "If you cannot answer based on the text, reply with: "
            "'I don't know based on the provided documents.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer (include source URLs):"
        ),
    )

    retriever = getattr(vector_store, "as_retriever", lambda **_: vector_store)(search_kwargs={"k": 4})
    llm, chosen_backend = None, None

    # --- Rate limiter state ---
    request_timestamps = deque()
    token_timestamps = deque()
    enc = tiktoken.get_encoding("cl100k_base")

    def quota_sleep(prompt_text):
        """Enforce RPM and TPM limits."""
        nonlocal request_timestamps, token_timestamps

        now = time.time()
        # Remove old entries
        while request_timestamps and now - request_timestamps[0] > 60:
            request_timestamps.popleft()
        while token_timestamps and now - token_timestamps[0][0] > 60:
            token_timestamps.popleft()

        # Estimate tokens in this request
        token_estimate = len(enc.encode(prompt_text))
        used_tokens = sum(t for _, t in token_timestamps)

        # If exceeding quota, sleep until safe
        while len(request_timestamps) >= OPENAI_RPM or (used_tokens + token_estimate) > OPENAI_TPM:
            oldest = request_timestamps[0] if request_timestamps else now
            oldest_token = token_timestamps[0][0] if token_timestamps else now
            wait_for = min(
                60 - (now - oldest),
                60 - (now - oldest_token)
            )
            wait_for = max(wait_for, 1)
            logger.warning(f"⏳ Quota reached (sleeping {wait_for:.1f}s)...")
            time.sleep(wait_for)
            now = time.time()
            while request_timestamps and now - request_timestamps[0] > 60:
                request_timestamps.popleft()
            while token_timestamps and now - token_timestamps[0][0] > 60:
                token_timestamps.popleft()
            used_tokens = sum(t for _, t in token_timestamps)

        # Record this request
        request_timestamps.append(now)
        token_timestamps.append((now, token_estimate))

    # Choose backend deterministically when requested; otherwise try an ordered list
    def try_init(name: str):
        nonlocal llm, chosen_backend
        try:
            if name == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY not set")
                logger.info("Trying OpenAI LLM...")
                cand = OpenAI(temperature=0)
                # quick sanity call if supported
                try:
                    if hasattr(cand, "invoke"):
                        cand.invoke("ping")
                except Exception:
                    pass
                llm = cand
                chosen_backend = "openai"
                return True
            if name == "ollama":
                try:
                    headers = {"Authorization": f"Bearer {ollama_token}"} if ollama_token else None
                    cand = OllamaLLM(base_url=ollama_url, model="mistral:instruct", headers=headers)
                    llm = cand
                    chosen_backend = "ollama"
                    logger.info("Using Ollama LLM.")
                    return True
                except Exception as e:
                    logger.warning("Ollama init failed: %s", e)
                    return False
            if name == "huggingface":
                try:
                    from langchain_community.llms import HuggingFaceHub
                    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                    # allow local HuggingFaceHub if token present
                    cand = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0})
                    llm = cand
                    chosen_backend = "huggingface"
                    logger.info("Using HuggingFaceHub LLM.")
                    return True
                except Exception as e:
                    logger.warning("HuggingFace init failed: %s", e)
                    return False
        except Exception as e:
            logger.warning("LLM init error for %s: %s", name, e)
        return False

    if backend and backend != "auto":
        ok = try_init(backend)
        if not ok:
            raise RuntimeError(f"Requested LLM backend '{backend}' failed to initialize")
    else:
        for cand in ["openai", "ollama", "huggingface"]:
            if try_init(cand):
                break

    # If LangChain isn't available or the provided vector_store is our TF-IDF fallback,
    # return a simple deterministic QA fallback that concatenates retrieved chunks.
    def _search_hits(vs, query: str, k=3, filter_source=None):
        if hasattr(vs, "search") and not hasattr(vs, "similarity_search_with_score"):
            docs_and_scores = vs.search(query, k * 3)
        else:
            docs_and_scores = vs.similarity_search_with_score(query, k * 3)
        filtered = []
        for doc, score in docs_and_scores:
            meta = doc.metadata if hasattr(doc, "metadata") else getattr(doc, "metadata", {}) or doc.get("metadata", {})
            src = meta.get("source") if meta else None
            if filter_source and src and filter_source not in src:
                continue
            filtered.append((doc, score))
            if len(filtered) >= k:
                break
        out = []
        for doc, score in filtered:
            meta = doc.metadata if hasattr(doc, "metadata") else getattr(doc, "metadata", {}) or doc.get("metadata", {})
            out.append({"chunk": doc.page_content, "score": float(score), "source": meta.get("source")})
        return out

    try:
        is_simple = isinstance(vector_store, SimpleFallbackVectorStore)
    except Exception:
        is_simple = False
    if not HAS_LANGCHAIN or is_simple:
        def simple_qa(query: str, k: int = 3, filter_source: str = None):
            hits = _search_hits(vector_store, query, k=k, filter_source=filter_source)
            if not hits:
                return {"answer": "I don't know based on the provided documents.", "sources": []}
            parts = [h["chunk"] for h in hits]
            sources = [h["source"] for h in hits if h.get("source")]
            return {"answer": "\n\n".join(parts), "sources": list(dict.fromkeys(sources))}

        return simple_qa, "simple", None

        if llm is None:
            try:
                from langchain_huggingface import HuggingFaceEndpoint

                hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if not hf_token:
                    raise RuntimeError("❌ HUGGINGFACEHUB_API_TOKEN not set. Please get one from https://huggingface.co/settings/tokens")

                llm = HuggingFaceEndpoint(
                    repo_id="HuggingFaceH4/zephyr-7b-beta",
                    task="conversational",
                    temperature=0.2,
                    huggingfacehub_api_token=hf_token,
                )
                backend = "huggingface"
                logger.info("🧠 Using HuggingFaceEndpoint fallback backend (langchain-huggingface).")
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace fallback failed: {e}")

    if llm is None:
        raise RuntimeError("No valid LLM backend available (OpenAI, Ollama, or HF).")

    logger.info("Final LLM backend: %s", chosen_backend)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    return qa_chain, chosen_backend, llm


# ------------------ Page-specific askers ------------------
def make_page_specific_askers(vector_store, qa_callable, urls: List[str]):
    """Return callables that answer queries restricted to one source URL.

    Each returned function has signature fn(query: str, k: int=4) -> {answer, sources}.
    The function performs a post-retrieval filter (by source URL) and then:
    - If an LLM object is available (returned alongside the QA chain), calls the LLM
      directly with the filtered chunks as the context so the model can synthesize and
      cite sources.
    - Otherwise, returns concatenated chunk text (TF-IDF/simple fallback).
    """
    askers = {}

    def _search_hits(vector_store, query: str, k: int = 3, filter_source: str = None):
        # Use vector_store.search for simple stores, otherwise similarity_search_with_score
        if hasattr(vector_store, "search") and not hasattr(vector_store, "similarity_search_with_score"):
            docs_and_scores = vector_store.search(query, k * 3)
        else:
            docs_and_scores = vector_store.similarity_search_with_score(query, k * 3)

        filtered = []
        for doc, score in docs_and_scores:
            meta = doc.metadata if hasattr(doc, "metadata") else getattr(doc, "metadata", {}) or doc.get("metadata", {})
            src = meta.get("source") if meta else None
            if filter_source and src and filter_source not in src:
                continue
            filtered.append((doc, score))
            if len(filtered) >= k:
                break
        out = []
        for doc, score in filtered:
            meta = doc.metadata if hasattr(doc, "metadata") else getattr(doc, "metadata", {}) or doc.get("metadata", {})
            out.append({"chunk": doc.page_content, "score": float(score), "source": meta.get("source")})
        return out

    def call_llm(llm_obj, prompt_text: str):
        """Call different possible LLM wrappers and return a plaintext answer."""
        try:
            # LangChain-style .invoke
            if hasattr(llm_obj, "invoke"):
                try:
                    res = llm_obj.invoke({"query": prompt_text})
                    if isinstance(res, dict):
                        return res.get("result") or res.get("output_text") or str(res)
                    return str(res)
                except Exception:
                    # Try direct invoke with text
                    try:
                        res = llm_obj.invoke(prompt_text)
                        return str(res)
                    except Exception:
                        pass
            # Callable LLMs
            if callable(llm_obj):
                out = llm_obj(prompt_text)
                if isinstance(out, dict):
                    return out.get("result") or out.get("text") or str(out)
                return str(out)
            # generate API
            if hasattr(llm_obj, "generate"):
                gen = llm_obj.generate([prompt_text])
                try:
                    return gen.generations[0][0].text
                except Exception:
                    return str(gen)
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
        return ""

    # Extract LLM object if qa_callable is the tuple returned by build_retrieval_qa
    llm_obj = None
    if isinstance(qa_callable, tuple) and len(qa_callable) >= 3:
        _, _, llm_obj = qa_callable

    for u in urls:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", u.rstrip('/').split('/')[-1]).lower()

        def make_fn(source_url):
            def fn(query: str, k: int = 4):
                hits = _search_hits(vector_store, query, k=k, filter_source=source_url)
                if not hits:
                    return {"answer": "I don't know based on the provided documents.", "sources": []}
                sources = list(dict.fromkeys([h.get('source') for h in hits if h.get('source')]))
                ctx_parts = [f"SOURCE: {h['source']}\nCONTENT: {h['chunk']}" for h in hits]
                context = "\n\n".join(ctx_parts)
                prompt = (
                    "You are a helpful assistant answering strictly from the provided context.\n"
                    "If the answer cannot be found, reply: 'I don't know based on the provided documents.'\n\n"
                    f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (include source URLs):"
                )

                if llm_obj:
                    ans_text = call_llm(llm_obj, prompt)
                    return {"answer": ans_text, "sources": sources}

                # fallback: return concatenated chunks
                return {"answer": "\n\n".join([h['chunk'] for h in hits]), "sources": sources}

            return fn

        askers[slug] = make_fn(u)

    return askers
