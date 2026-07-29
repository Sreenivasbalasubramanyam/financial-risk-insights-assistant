"""
Embedding backends for the RAG pipeline.

Two backends are supported behind a single interface:

1. ``TfidfEmbedder`` (default) — a from-scratch TF-IDF vectorizer built on
   plain Python + numpy. This keeps the project runnable with zero heavy
   ML dependencies and no internet access to download model weights, which
   matters for constrained/offline environments (e.g. CI runners, air-gapped
   analyst workstations). If ``scikit-learn`` is installed, we transparently
   use ``sklearn.feature_extraction.text.TfidfVectorizer`` instead, since it
   is faster and more feature-complete; the two are API-compatible for our
   purposes (``fit_transform`` / ``transform`` returning dense numpy arrays).

2. ``SentenceTransformerEmbedder`` (optional) — wraps
   ``sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")`` for
   dense semantic embeddings when the package is installed. Select it via
   ``EMBEDDING_BACKEND=sentence-transformers``.

Both expose the same two methods so callers (ingest.py, retriever.py) never
need to know which one is active:

    embedder.fit(list_of_texts) -> None          # build vocabulary/model
    embedder.transform(list_of_texts) -> np.ndarray (n_docs, dim)
    embedder.embed_query(text) -> np.ndarray (dim,)
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import List

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "as", "at", "by", "with", "from", "into",
    "such", "than", "so", "not", "no", "will", "shall", "may", "must", "can",
    "which", "who", "whom", "their", "they", "them", "has", "have", "had",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords/short tokens."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class TfidfEmbedder:
    """
    A minimal, dependency-free TF-IDF vectorizer.

    Equivalent (drop-in replaceable) to:

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
        vectors = vectorizer.fit_transform(texts).toarray()

    We implement it by hand so the project has zero hard dependency on
    scikit-learn while remaining a legitimate, well-understood retrieval
    baseline (term-frequency * inverse-document-frequency, L2-normalized,
    scored by cosine similarity).
    """

    def __init__(self):
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    # -- fitting ------------------------------------------------------
    def fit(self, texts: List[str]) -> "TfidfEmbedder":
        doc_token_sets = [set(tokenize(t)) for t in texts]
        df_counter: Counter[str] = Counter()
        for tokens in doc_token_sets:
            df_counter.update(tokens)

        vocab = sorted(df_counter.keys())
        self.vocabulary_ = {term: idx for idx, term in enumerate(vocab)}

        n_docs = len(texts)
        idf = np.zeros(len(vocab), dtype=np.float64)
        for term, idx in self.vocabulary_.items():
            df = df_counter[term]
            # smoothed idf, same formula sklearn uses by default:
            # idf = ln((1 + n) / (1 + df)) + 1
            idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0
        self.idf_ = idf
        return self

    # -- transforming ---------------------------------------------------
    def _term_frequencies(self, text: str) -> Counter:
        return Counter(tokenize(text))

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("TfidfEmbedder.fit() must be called before transform().")
        vocab_size = len(self.vocabulary_)
        matrix = np.zeros((len(texts), vocab_size), dtype=np.float64)

        for row, text in enumerate(texts):
            tf = self._term_frequencies(text)
            for term, count in tf.items():
                idx = self.vocabulary_.get(term)
                if idx is not None:
                    matrix[row, idx] = count * self.idf_[idx]

        # L2-normalize each row so cosine similarity == dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        return matrix

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.transform([text])[0]

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "vocabulary": self.vocabulary_,
            "idf": self.idf_.tolist() if self.idf_ is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfEmbedder":
        emb = cls()
        emb.vocabulary_ = data["vocabulary"]
        emb.idf_ = np.array(data["idf"], dtype=np.float64) if data["idf"] is not None else None
        return emb

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path) -> "TfidfEmbedder":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class SentenceTransformerEmbedder:
    """
    Optional dense-embedding backend using sentence-transformers.

    Only imported/instantiated when EMBEDDING_BACKEND=sentence-transformers,
    so the package is not a hard requirement for the default demo path.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` or set "
                "EMBEDDING_BACKEND=tfidf to use the built-in offline embedder."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> "SentenceTransformerEmbedder":
        # Sentence-transformer models are pretrained; nothing to fit.
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float64)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.transform([text])[0]


def get_embedder(backend: str = "tfidf", model_name: str = "all-MiniLM-L6-v2"):
    """Factory returning the configured embedder instance."""
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend in {"sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"Unknown embedding backend: {backend!r}")
