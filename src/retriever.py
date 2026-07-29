"""
Retriever: loads a persisted index (or builds one on the fly) and performs
similarity search, returning ranked chunks with source metadata.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from src.config import (
    DATA_DIR,
    INDEX_METADATA_PATH,
    INDEX_VECTORS_PATH,
    INDEX_VOCAB_PATH,
    settings,
)
from src.embeddings import TfidfEmbedder, get_embedder
from src.ingest import build_index
from src.vector_index import load_index


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_id: str
    chunk_index: int
    score: float


class Retriever:
    def __init__(self, embedder, index, chunk_metadata: List[dict]):
        self.embedder = embedder
        self.index = index
        self.chunk_metadata = chunk_metadata

    @classmethod
    def from_disk(cls) -> "Retriever":
        """Load a previously built index from data/index/. Raises FileNotFoundError
        if no index has been built yet (run `python -m src.ingest` first)."""
        if not (INDEX_METADATA_PATH.exists() and INDEX_VECTORS_PATH.exists()):
            raise FileNotFoundError(
                "No index found on disk. Run `python -m src.ingest` first to "
                "build data/index/."
            )
        metadata = json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
        embedding_backend = metadata.get("embedding_backend", settings.embedding_backend)
        vector_index_backend = metadata.get("vector_index_backend", settings.vector_index_backend)

        if embedding_backend == "tfidf" and INDEX_VOCAB_PATH.exists():
            embedder = TfidfEmbedder.load(INDEX_VOCAB_PATH)
        else:
            embedder = get_embedder(embedding_backend, model_name=settings.sentence_transformer_model)

        index = load_index(vector_index_backend, INDEX_VECTORS_PATH)
        return cls(embedder, index, metadata["chunks"])

    @classmethod
    def build_fresh(cls, data_dir=DATA_DIR) -> "Retriever":
        """Build the index in-memory (and persist it) from data_dir. Useful
        for tests that don't want to depend on a previously-run ingest step."""
        embedder, index, chunks = build_index(data_dir=data_dir)
        chunk_metadata = [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            for c in chunks
        ]
        return cls(embedder, index, chunk_metadata)

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k
        query_vector = self.embedder.embed_query(query)
        scores, indices = self.index.search(query_vector, k=top_k)

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunk_metadata):
                continue
            meta = self.chunk_metadata[idx]
            results.append(
                RetrievedChunk(
                    text=meta["text"],
                    source=meta["source"],
                    chunk_id=meta["chunk_id"],
                    chunk_index=meta["chunk_index"],
                    score=float(score),
                )
            )
        return results


def get_retriever() -> Retriever:
    """Convenience factory: load from disk if available, else build fresh."""
    try:
        return Retriever.from_disk()
    except FileNotFoundError:
        return Retriever.build_fresh()
