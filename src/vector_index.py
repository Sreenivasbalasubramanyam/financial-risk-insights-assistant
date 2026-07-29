"""
Vector index backends.

``NumpyCosineIndex`` is a small in-memory cosine-similarity index built on
plain numpy. It is the default backend because it has zero extra
dependencies, is trivial to reason about, and is plenty fast for the
document volumes a demo/portfolio project like this deals with (dozens to
low thousands of chunks). This is a deliberate, documented lightweight
alternative to FAISS for environments where installing `faiss-cpu` is slow,
version-fragile, or unavailable (e.g. some ARM/CI/sandboxed environments).

``FaissIndex`` wraps ``faiss.IndexFlatIP`` (inner product on L2-normalized
vectors == cosine similarity) when the ``faiss-cpu`` package is installed.
Both expose the same three methods so ``retriever.py`` can use either one
interchangeably:

    index.add(vectors: np.ndarray) -> None
    index.search(query_vector: np.ndarray, k: int) -> (scores, indices)
    index.save(path) / index.load(path)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyCosineIndex:
    """In-memory cosine-similarity search over L2-normalized vectors."""

    def __init__(self):
        self._vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float64)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query_vector: np.ndarray, k: int = 3):
        if self._vectors is None or len(self._vectors) == 0:
            return np.array([]), np.array([], dtype=int)

        query_vector = np.asarray(query_vector, dtype=np.float64)
        q_norm = np.linalg.norm(query_vector)
        if q_norm > 0:
            query_vector = query_vector / q_norm

        # Vectors are assumed pre-normalized (embeddings.py L2-normalizes),
        # so dot product == cosine similarity.
        scores = self._vectors @ query_vector
        k = min(k, len(scores))
        top_idx = np.argsort(-scores)[:k]
        return scores[top_idx], top_idx

    def save(self, path) -> None:
        np.savez(path, vectors=self._vectors)

    @classmethod
    def load(cls, path) -> "NumpyCosineIndex":
        idx = cls()
        data = np.load(path)
        idx._vectors = data["vectors"]
        return idx

    def __len__(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)


class FaissIndex:
    """FAISS-backed flat inner-product index (cosine similarity via L2-normalized vectors)."""

    def __init__(self, dim: int | None = None):
        try:
            import faiss  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "faiss-cpu is not installed. Install it with `pip install "
                "faiss-cpu`, or set VECTOR_INDEX_BACKEND=numpy to use the "
                "built-in NumpyCosineIndex instead."
            ) from exc
        self._faiss = faiss
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim) if dim else None

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self._index is None:
            self._dim = vectors.shape[1]
            self._index = self._faiss.IndexFlatIP(self._dim)
        self._index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int = 3):
        query_vector = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )
        scores, indices = self._index.search(query_vector, k)
        return scores[0], indices[0]

    def save(self, path) -> None:
        self._faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path) -> "FaissIndex":
        import faiss

        idx = cls.__new__(cls)
        idx._faiss = faiss
        idx._index = faiss.read_index(str(path))
        idx._dim = idx._index.d
        return idx

    def __len__(self) -> int:
        return 0 if self._index is None else self._index.ntotal


def get_index(backend: str = "numpy", dim: int | None = None):
    """Factory returning an empty index of the configured backend."""
    if backend == "numpy":
        return NumpyCosineIndex()
    if backend == "faiss":
        return FaissIndex(dim=dim)
    raise ValueError(f"Unknown vector index backend: {backend!r}")


def load_index(backend: str, path: Path):
    if backend == "numpy":
        return NumpyCosineIndex.load(path)
    if backend == "faiss":
        return FaissIndex.load(path)
    raise ValueError(f"Unknown vector index backend: {backend!r}")
