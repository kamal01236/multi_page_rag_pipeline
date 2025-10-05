"""
rag_pipeline.py (updated)
Supports OpenAI, HuggingFace, or TF-IDF embeddings/LLM for RAG pipeline.
"""

import os, re, random, logging
from typing import List, Optional, Tuple, Any
import requests
from bs4 import BeautifulSoup

# LangChain
try:
    from langchain.docstore.document import Document
    from langchain_openai import OpenAIEmbeddings, OpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False
    try:
        from langchain.prompts import PromptTemplate
    except Exception:
        PromptTemplate = None

# Local fallback
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
    for tag in content.find_all(["table","style","script","sup","img","aside"]):
        tag.decompose()
    stop_headings = {"References","External links","See also","Further reading"}
    paragraphs = []
    for elem in content.find_all(["h2","h3","p","li"]):
        if elem.name in ("h2","h3"):
            heading = re.sub(r"\[.*?\]","",elem.get_text(" ",strip=True))
            if any(h in heading for h in stop_headings):
                break
            continue
        text = elem.get_text(" ",strip=True)
        if not text: continue
        text = re.sub(r"\[\d+\]","",text)
        text = re.sub(r"\[citation needed\]","",text,flags=re.I)
        text = re.sub(r"\[.*?\]","",text)
        text = re.sub(r"\s+"," ",text).strip()
        if len(text) > 30:
            paragraphs.append(text)
    return title_text, "\n\n".join(paragraphs)

# ------------------ Chunking ------------------
def randomized_chunks(text: str, min_size=MIN_CHUNK, max_size=MAX_CHUNK, overlap=OVERLAP, seed=None):
    if seed is not None: random.seed(seed)
    chunks, pos, n = [], 0, len(text)
    while pos < n:
        size = random.randint(min_size,max_size)
        chunk = text[pos:pos+size].strip()
        if not chunk: break
        chunks.append(chunk)
        pos += max(1,size-overlap)
        if len(chunks) > 10000: break
    return chunks

# ------------------ Vector Store ------------------
class SimpleFallbackVectorStore:
    def __init__(self, docs: List[Any]):
        if not HAS_SKLEARN: raise RuntimeError("Need scikit-learn")
        self.docs = docs
        texts = [d.page_content for d in docs]
        self.tfidf = TfidfVectorizer().fit(texts+["placeholder"])
        self.vectors = self.tfidf.transform(texts).toarray()
    def search(self, query, k=3):
        qv = self.tfidf.transform([query]).toarray()[0]
        sims = (self.vectors @ qv)/((np.linalg.norm(self.vectors,axis=1)*(np.linalg.norm(qv)+1e-9))+1e-9)
        idxs = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in idxs]

def build_vectorstore(docs: List[Any]):
    """Build the vector store using OpenAI if API key is available; otherwise fall back."""

    import os
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import time

    if os.getenv("OPENAI_API_KEY"):
        logger.info("✅ OpenAI API key detected — using OpenAI embeddings via FAISS.")
        try:
            emb = OpenAIEmbeddings(model="text-embedding-3-small")
            # Slow embedding creation to avoid rate limits
            for i in range(0, len(docs), 20):  # 20 docs at a time
                batch = docs[i:i+20]
                store = FAISS.from_documents(batch, emb) if i == 0 else store.merge_from(FAISS.from_documents(batch, emb))
                time.sleep(0.5)

            return store, "faiss-openai"
        except Exception as e:
            logger.error("❌ OpenAI embedding failed: %s", e)

    # HuggingFace fallback
    try:
        logger.info("⚙️  Using HuggingFace sentence-transformers fallback.")
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        store = FAISS.from_documents(docs, emb)
        return store, "faiss-hf"
    except Exception as e:
        logger.error("❌ HuggingFace embedding failed: %s", e)

    # TF-IDF fallback (always available)
    logger.warning("⚠️  Falling back to simple TF-IDF embeddings (no LLM).")

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

    return SimpleFallbackVectorStore(docs), "tfidf"


def detect_backend() -> str:
    """Return a short string describing which embedding/LLM backend would be used.

    This function is deterministic based on installed optional packages and environment variables.
    It is useful for tests and for logging from the CLI.
    """
    if HAS_LANGCHAIN:
        if os.getenv("OPENAI_API_KEY"):
            return "faiss-openai"
        return "faiss-hf"
    if HAS_SKLEARN:
        return "tfidf"
    return "none"

# ------------------ Search ------------------
def search_db(vector_store, query, k=3, filter_source=None):
    if isinstance(vector_store, SimpleFallbackVectorStore):
        hits = vector_store.search(query,k*3)
    else:
        hits = vector_store.similarity_search_with_score(query,k*3)
    filtered=[]
    for doc,score in hits:
        meta = doc.metadata if hasattr(doc,"metadata") else doc.get("metadata",{})
        src = meta.get("source") if meta else None
        if filter_source and src and filter_source not in src: continue
        filtered.append((doc,score))
        if len(filtered)>=k: break
    out=[]; 
    for doc,score in filtered:
        meta = doc.metadata if hasattr(doc,"metadata") else doc.get("metadata",{})
        out.append({"chunk":doc.page_content,"score":float(score),"source":meta.get("source")})
    return out

# ------------------ RetrievalQA ------------------
from langchain.chains import RetrievalQA

def build_retrieval_qa(vector_store):
    """Build RetrievalQA chain with OpenAI or HuggingFace backend."""
    from langchain.prompts import PromptTemplate
    from langchain_openai import OpenAI
    from langchain.chains import RetrievalQA

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant answering based strictly on the provided context.\n"
            "If you cannot answer based on the text, reply with: 'I don't know based on the provided documents.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer (include source URLs):"
        ),
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = OpenAI(temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    return qa_chain, "langchain"



def make_page_specific_askers(vector_store, qa_callable, urls: List[str]):
    """Create page-specific askers based on the provided URLs.

    Returns a dict mapping a safe slug (e.g. 'quantum_computing') to a function fn(query) -> {answer,sources}.
    """
    askers = {}
    for u in urls:
        slug = re.sub(r"[^0-9a-zA-Z]+","_", u.rstrip('/').split('/')[-1]).lower()
        def make_fn(source_url):
            return lambda query, k=4: qa_callable(query, k=k, filter_source=source_url)
        askers[slug] = make_fn(u)
    return askers
