from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from .rag_pipeline import demo_pipeline, build_vectorstore, build_retrieval_qa, make_page_specific_askers
from .config import CONFIG

app = FastAPI(title="Multi-Page RAG Pipeline API")
logger = logging.getLogger(__name__)


class QARequest(BaseModel):
    urls: List[str]
    query: str
    emb_backend: Optional[str] = None
    llm_backend: Optional[str] = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    # In a real deployment we'd check model availability / vector store readiness
    return {"ready": True}


@app.post("/qa")
async def qa(req: QARequest):
    try:
        docs, store = None, None
        # Build pipeline (small demo)
        docs, store = (None, None)
        # Build docs and vectorstore via demo_pipeline for simplicity
        result = demo_pipeline(urls=req.urls, seed=42, backend=(req.emb_backend or CONFIG.emb_backend or 'auto'))
        qa_chain = result['qa']
        # Use page-specific askers or general QA
        res = qa_chain.invoke({"query": req.query}) if hasattr(qa_chain, 'invoke') else qa_chain(req.query)
        return {"result": res}
    except Exception as e:
        logger.exception("QA call failed")
        raise HTTPException(status_code=500, detail=str(e))
