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
import time, tiktoken
from collections import deque

# ------------------ RetrievalQA ------------------
def build_retrieval_qa(vector_store, ollama_url="http://localhost:11434", ollama_token=None):
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
    llm, backend = None, None

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

    # --- Backend setup ---
    force_ollama = os.getenv("FORCE_OLLAMA", "0").strip() == "1"
    ollama_token = os.getenv("OLLAMA_TOKEN")
    logger.info(f"FORCE_OLLAMA={force_ollama}")

    if force_ollama:
        logger.info("🚀 FORCE_OLLAMA=1 — using local Ollama backend.")
        headers = {"Authorization": f"Bearer {ollama_token}"} if ollama_token else None
        llm = OllamaLLM(base_url=ollama_url, model="mistral:instruct", headers=headers)
        backend = "ollama"
    else:
        if os.getenv("OPENAI_API_KEY"):
            try:
                logger.info("🔷 Trying OpenAI backend (with quota limiter)...")

                # Wrap the LangChain OpenAI LLM to include rate limiting + retry logic
                from langchain_openai import OpenAI as LCOpenAI
                from openai import RateLimitError

                class QuotaSafeOpenAI(LCOpenAI):
                    def invoke(self, prompt_text, *args, **kwargs):
                        quota_sleep(prompt_text)
                        for attempt in range(5):
                            try:
                                return super().invoke(prompt_text, *args, **kwargs)
                            except RateLimitError:
                                wait = 2 ** attempt + random.uniform(0, 1)
                                logger.warning(f"⚠️ Rate limit — retrying in {wait:.1f}s...")
                                time.sleep(wait)
                            except Exception as e:
                                if "429" in str(e):
                                    wait = 2 ** attempt + random.uniform(0, 1)
                                    logger.warning(f"⚠️ 429 error — retrying in {wait:.1f}s...")
                                    time.sleep(wait)
                                else:
                                    raise
                        raise RuntimeError("❌ Too many retries due to rate limits")

                llm = QuotaSafeOpenAI(temperature=0)
                _ = llm.invoke("ping")  # sanity check
                backend = "openai"
            except Exception as e:
                logger.warning(f"⚠️ OpenAI failed ({type(e).__name__}): {e}")
                llm = None

        if llm is None:
            try:
                logger.info("🟢 Trying local Ollama backend...")
                r = requests.get(f"{ollama_url}/api/tags", timeout=2)
                if r.ok:
                    headers = {"Authorization": f"Bearer {ollama_token}"} if ollama_token else None
                    llm = OllamaLLM(base_url=ollama_url, model="mistral:instruct", headers=headers)
                    backend = "ollama"
            except Exception as e:
                logger.warning(f"⚠️ Ollama not reachable: {e}")

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
        raise RuntimeError("❌ No valid LLM backend available (OpenAI, Ollama, or HF).")

    logger.info(f"✅ Final LLM backend: {backend}")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    return qa_chain, backend, llm


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
