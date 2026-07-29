"""
End-to-end test: run answer_query() against the real synthetic documents in
data/policy_documents/ and confirm we get a non-empty, citation-grounded
answer using the default (extractive, no-API-key) backend.
"""
import os

import pytest

os.environ.setdefault("LLM_BACKEND", "extractive")

from src.ingest import build_index  # noqa: E402
from src.rag_pipeline import answer_query, reset_retriever_cache  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _ensure_index_built():
    """Build the real index from data/policy_documents/ once for this test module."""
    build_index()
    reset_retriever_cache()
    yield


def test_answer_query_counterparty_exposure_question():
    result = answer_query("What is the policy on counterparty exposure limits?")

    assert result["question"]
    assert result["answer"].strip() != ""
    assert result["backend"] == "extractive"
    assert len(result["citations"]) > 0

    # At least one citation should point at the counterparty exposure policy doc.
    sources = [c["source"] for c in result["citations"]]
    assert any("counterparty_exposure" in s for s in sources)

    # The answer text should itself contain an inline citation tag.
    assert "risk_policy_counterparty_exposure.txt" in result["answer"]


def test_answer_query_fraud_question_returns_relevant_citation():
    result = answer_query("How are fraudulent transactions detected and scored?")

    assert result["answer"].strip() != ""
    assert len(result["citations"]) > 0
    sources = [c["source"] for c in result["citations"]]
    assert any("fraud" in s for s in sources)


def test_answer_query_raises_on_empty_question():
    with pytest.raises(ValueError):
        answer_query("")
