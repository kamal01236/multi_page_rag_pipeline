from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import threading

from src.rag_pipeline import (
    build_from_urls,
    search_db,
    make_page_specific_askers,
    build_retrieval_qa,     # ✅ added import
)

logger = logging.getLogger("multi_page_rag_server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Multi-Page RAG Pipeline API")

# ------------------- Global In-Memory State -------------------
STATE = {
    "docs": [],
    "vector_store": None,
    "qa_callable": None,
    "askers": {},  # slug -> callable
    "metadata": [],
    "lock": threading.Lock(),
    "backend": "auto",
}

# ------------------- Request Models -------------------
class IngestRequest(BaseModel):
    urls: List[str]
    seed: Optional[int] = None
    backend: Optional[str] = "auto"
    overwrite: Optional[bool] = False


class QARequest(BaseModel):
    query: str
    k: Optional[int] = 3
    filter_source: Optional[str] = None


class BatchQARequest(BaseModel):
    queries: List[str]
    k: Optional[int] = 3


# ------------------- Health Check -------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "backend": STATE.get("backend")}


# ------------------- Ingest URLs -------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    if not build_from_urls:
        raise HTTPException(status_code=500, detail="Ingest function not available in rag_pipeline module")

    with STATE["lock"]:
        try:
            docs, store = build_from_urls(req.urls, seed=req.seed)
        except Exception as e:
            logger.exception("❌ Ingestion failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

        STATE["docs"] = docs
        STATE["vector_store"] = store
        STATE["backend"] = req.backend or STATE["backend"]

        # ------------------- Build Retrieval QA -------------------
        try:
            qa_callable, backend_name, llm_obj = build_retrieval_qa(store, backend=req.backend)
            STATE["qa_callable"] = qa_callable
            STATE["backend"] = backend_name
        except Exception as e:
            logger.warning("⚠️ Falling back to simple QA callable: %s", e)
            STATE["qa_callable"] = None

        # ------------------- Build Page-Specific Askers -------------------
        try:
            # make_page_specific_askers expects the vector_store and a list of canonical URLs
            askers = make_page_specific_askers(store, req.urls)
            STATE["askers"] = askers
        except Exception as e:
            logger.warning("⚠️ Could not create page-specific askers: %s", e)
            STATE["askers"] = {}

    return {
        "message": "✅ Ingestion completed.",
        "pages_indexed": len(req.urls),
        "total_chunks": len(STATE["docs"]),
        "backend": STATE["backend"],
        "askers_available": len(STATE["askers"]),
    }


# ------------------- List Available Pages -------------------
@app.get("/pages")
def list_pages():
    if not STATE["docs"]:
        return {"count": 0, "pages": []}
    sources = {
        getattr(d, "metadata", {}).get("source")
        for d in STATE["docs"]
        if getattr(d, "metadata", {}).get("source")
    }
    return {"count": len(sources), "pages": sorted(list(sources))}


# ------------------- General QA -------------------
@app.post("/qa")
def qa(req: QARequest):
    if not STATE["vector_store"]:
        raise HTTPException(status_code=400, detail="No vector store available. Run /ingest first.")

    qa_callable = STATE.get("qa_callable")

    # Prefer full LLM QA chain if available
    if qa_callable:
        try:
            res = qa_callable(req.query, k=req.k, filter_source=req.filter_source)
            return {
                "query": req.query,
                "answer": res.get("answer"),
                "sources": res.get("sources"),
                "backend": STATE["backend"],
            }
        except Exception as e:
            logger.warning("QA callable failed: %s", e)

    # Fallback: use pure retrieval (no LLM)
    try:
        hits = search_db(STATE["vector_store"], req.query, k=req.k, filter_source=req.filter_source)
        if not hits:
            return {
                "query": req.query,
                "answer": "I don't know based on the provided documents.",
                "sources": [],
                "backend": STATE["backend"],
            }
        answer = "\n\n".join([h["chunk"] for h in hits])
        sources = list({h["source"] for h in hits if h.get("source")})
        confidence = sum(h["score"] for h in hits) / len(hits)
        return {
            "query": req.query,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "backend": STATE["backend"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {e}")


# ------------------- Page-Specific QA -------------------
@app.post("/qa/{slug}")
def qa_page(slug: str, req: QARequest):
    askers = STATE.get("askers", {})
    if slug not in askers:
        return {"query": req.query, "answer": f"No asker available for slug '{slug}'", "sources": []}
    try:
        res = askers[slug](req.query, k=req.k)
        return {
            "query": req.query,
            "answer": res.get("answer"),
            "sources": res.get("sources"),
            "backend": STATE["backend"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Page-specific QA failed: {e}")


# ------------------- Batch QA -------------------
@app.post("/batch_qa")
def batch_qa(req: BatchQARequest):
    if not STATE["vector_store"]:
        raise HTTPException(status_code=400, detail="No vector store available. Run /ingest first.")

    results = []
    for q in req.queries:
        try:
            res = qa(QARequest(query=q, k=req.k))
            results.append(res)
        except Exception as e:
            results.append({"query": q, "error": str(e)})

    return {"count": len(results), "results": results}
