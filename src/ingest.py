"""
Ingestion pipeline: load documents -> chunk -> embed -> build & persist index.

Run as a script:

    python -m src.ingest

This walks `data/policy_documents/`, splits each file into overlapping
chunks (src/chunking.py), embeds the chunks (src/embeddings.py, default
TF-IDF backend), builds a similarity index (src/vector_index.py, default
in-memory numpy cosine index), and writes everything to `data/index/` so
`src/retriever.py` can load it without re-embedding on every query.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from src.chunking import Chunk, chunk_document
from src.config import (
    DATA_DIR,
    INDEX_METADATA_PATH,
    INDEX_VECTORS_PATH,
    INDEX_VOCAB_PATH,
    settings,
)
from src.embeddings import TfidfEmbedder, get_embedder
from src.vector_index import get_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_documents(data_dir: Path = DATA_DIR) -> List[tuple[str, str]]:
    """Return list of (filename, raw_text) for every .txt file in data_dir."""
    docs = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append((path.name, text))
    return docs


def build_chunks(documents: List[tuple[str, str]]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for filename, text in documents:
        chunks = chunk_document(
            text,
            source=filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        all_chunks.extend(chunks)
        logger.info("  %-45s -> %d chunk(s)", filename, len(chunks))
    return all_chunks


def build_index(
    data_dir: Path = DATA_DIR,
    embedding_backend: str | None = None,
    vector_index_backend: str | None = None,
):
    """Full ingest pipeline. Returns (embedder, index, chunks) and persists to disk."""
    embedding_backend = embedding_backend or settings.embedding_backend
    vector_index_backend = vector_index_backend or settings.vector_index_backend

    logger.info("Loading documents from %s", data_dir)
    documents = load_documents(data_dir)
    if not documents:
        raise FileNotFoundError(f"No .txt documents found in {data_dir}")
    logger.info("Loaded %d document(s)", len(documents))

    logger.info("Chunking documents (chunk_size=%d, overlap=%d)...", settings.chunk_size, settings.chunk_overlap)
    chunks = build_chunks(documents)
    logger.info("Produced %d total chunk(s)", len(chunks))

    texts = [c.text for c in chunks]

    logger.info("Embedding chunks with backend=%s", embedding_backend)
    embedder = get_embedder(embedding_backend, model_name=settings.sentence_transformer_model)
    vectors = embedder.fit_transform(texts)
    logger.info("Embedding matrix shape: %s", vectors.shape)

    logger.info("Building vector index with backend=%s", vector_index_backend)
    index = get_index(vector_index_backend, dim=vectors.shape[1])
    index.add(vectors)

    _persist(embedder, index, chunks, embedding_backend, vector_index_backend)
    logger.info("Index build complete. %d chunks indexed.", len(chunks))
    return embedder, index, chunks


def _persist(embedder, index, chunks: List[Chunk], embedding_backend: str, vector_index_backend: str) -> None:
    # Chunk metadata (source file, chunk id, raw text) so retrieval can
    # return human-readable citations.
    metadata = {
        "embedding_backend": embedding_backend,
        "vector_index_backend": vector_index_backend,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            for c in chunks
        ],
    }
    INDEX_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if isinstance(embedder, TfidfEmbedder):
        embedder.save(INDEX_VOCAB_PATH)

    index.save(INDEX_VECTORS_PATH)


if __name__ == "__main__":
    build_index()
