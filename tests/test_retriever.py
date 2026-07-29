"""
Unit test for the retrieval layer: build a tiny in-memory index from a
handful of synthetic chunks and confirm the retriever ranks the most
lexically/semantically relevant chunk first.
"""
from src.embeddings import get_embedder
from src.retriever import Retriever
from src.vector_index import get_index


SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc_a::chunk_0",
        "source": "doc_a.txt",
        "chunk_index": 0,
        "text": (
            "Counterparty exposure limits are capped at 8 percent of Tier 1 "
            "capital for investment-grade counterparties and 2 percent for "
            "below investment-grade counterparties."
        ),
    },
    {
        "chunk_id": "doc_b::chunk_0",
        "source": "doc_b.txt",
        "chunk_index": 0,
        "text": (
            "The Fraud Detection Engine scores every transaction from 0 to "
            "100 using device fingerprinting and velocity checks, auto-"
            "declining scores above 85."
        ),
    },
    {
        "chunk_id": "doc_c::chunk_0",
        "source": "doc_c.txt",
        "chunk_index": 0,
        "text": (
            "Settlement risk arises when a counterparty fails to deliver its "
            "side of a trade after the bank has already delivered its side, "
            "exposing the bank to principal risk."
        ),
    },
]


def _build_test_retriever() -> Retriever:
    texts = [c["text"] for c in SAMPLE_CHUNKS]
    embedder = get_embedder("tfidf")
    vectors = embedder.fit_transform(texts)

    index = get_index("numpy")
    index.add(vectors)

    return Retriever(embedder=embedder, index=index, chunk_metadata=SAMPLE_CHUNKS)


def test_retrieval_ranks_most_relevant_chunk_first():
    retriever = _build_test_retriever()

    results = retriever.retrieve("What are the counterparty exposure limits?", top_k=3)

    assert len(results) > 0
    top_result = results[0]
    assert top_result.source == "doc_a.txt"
    assert "exposure" in top_result.text.lower()
    # Scores should be sorted descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_returns_fraud_chunk_for_fraud_question():
    retriever = _build_test_retriever()

    results = retriever.retrieve("How does the fraud detection engine score transactions?", top_k=1)

    assert len(results) == 1
    assert results[0].source == "doc_b.txt"


def test_retrieve_respects_top_k():
    retriever = _build_test_retriever()
    results = retriever.retrieve("settlement risk counterparty", top_k=2)
    assert len(results) == 2
