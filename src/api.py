"""
FastAPI service exposing the RAG pipeline for analyst tooling / internal
knowledge retrieval.

Run locally:

    uvicorn src.api:app --reload --port 8000

Endpoints:

    GET  /health          -> {"status": "ok"}
    POST /query           -> {"question": ..., "answer": ..., "backend": ..., "citations": [...]}
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.rag_pipeline import answer_query

app = FastAPI(
    title="Financial Risk Insights Assistant",
    description=(
        "Retrieval-augmented generation API for financial risk policy, "
        "fraud detection, AML, and settlement-risk knowledge lookup."
    ),
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question for the assistant.")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve. Defaults to server config.")


class Citation(BaseModel):
    source: str
    chunk_id: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    backend: str
    citations: list[Citation]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_backend": settings.effective_llm_backend(),
        "embedding_backend": settings.embedding_backend,
        "vector_index_backend": settings.vector_index_backend,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = answer_query(request.question, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Index not built yet: {exc}. Run `python -m src.ingest` first.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return QueryResponse(**result)
