"""
Central configuration for the Financial Risk Insights Assistant.

All settings are environment-variable-overridable so the same code can run
locally (fully offline, no API keys) or be pointed at an OpenAI-backed
deployment without touching any source files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "policy_documents"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_METADATA_PATH = INDEX_DIR / "chunks_metadata.json"
INDEX_VECTORS_PATH = INDEX_DIR / "vectors.npz"
INDEX_VOCAB_PATH = INDEX_DIR / "vocabulary.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 700))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))

    # --- Retrieval ---
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 3))

    # --- Embedding backend: "tfidf" (default, no heavy deps) or "sentence-transformers" ---
    embedding_backend: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_BACKEND", "tfidf")
    )
    sentence_transformer_model: str = field(
        default_factory=lambda: os.environ.get(
            "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
        )
    )

    # --- Vector index backend: "numpy" (default) or "faiss" ---
    vector_index_backend: str = field(
        default_factory=lambda: os.environ.get("VECTOR_INDEX_BACKEND", "numpy")
    )

    # --- LLM / answer-generation backend: "extractive" (default, no API key) or "openai" ---
    llm_backend: str = field(
        default_factory=lambda: os.environ.get("LLM_BACKEND", "extractive")
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    )

    # --- Answer formatting ---
    max_context_chars: int = field(
        default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 2000)
    )

    def effective_llm_backend(self) -> str:
        """Resolve to 'openai' only if explicitly requested AND a key is present."""
        if self.llm_backend == "openai" and self.openai_api_key:
            return "openai"
        return "extractive"


settings = Settings()
