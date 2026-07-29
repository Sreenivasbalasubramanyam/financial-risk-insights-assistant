"""
End-to-end RAG orchestration: retrieve relevant chunks, generate a
citation-grounded answer, and package the result for API/CLI consumers.
"""
from __future__ import annotations

from typing import Optional

from src.config import settings
from src.llm_backend import get_llm_backend
from src.retriever import Retriever, get_retriever

_retriever_singleton: Optional[Retriever] = None


def _get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = get_retriever()
    return _retriever_singleton


def reset_retriever_cache() -> None:
    """Force the next call to answer_query() to reload the retriever from
    disk. Useful after re-running ingest, or in tests."""
    global _retriever_singleton
    _retriever_singleton = None


def answer_query(question: str, top_k: int | None = None) -> dict:
    """
    Run the full RAG pipeline for a single question.

    Returns:
        {
            "question": str,
            "answer": str,
            "backend": str,               # "extractive" or "openai"
            "citations": [
                {"source": str, "chunk_id": str, "chunk_index": int, "score": float}
            ],
        }
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    retriever = _get_retriever()
    chunks = retriever.retrieve(question, top_k=top_k or settings.top_k)

    llm = get_llm_backend()
    answer_text = llm.generate(question, chunks)

    citations = [
        {
            "source": c.source,
            "chunk_id": c.chunk_id,
            "chunk_index": c.chunk_index,
            "score": round(c.score, 4),
        }
        for c in chunks
    ]

    return {
        "question": question,
        "answer": answer_text,
        "backend": llm.name,
        "citations": citations,
    }


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "What is the policy on counterparty exposure limits?"
    result = answer_query(q)
    print(json.dumps(result, indent=2))
