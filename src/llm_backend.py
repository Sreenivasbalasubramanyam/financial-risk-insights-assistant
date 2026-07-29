"""
Pluggable answer-generation backends.

Two backends implement the same interface, `generate(question, chunks) -> str`:

* ``ExtractiveAnswerBackend`` (default, no API key required) — builds a
  citation-grounded answer by selecting and lightly stitching together the
  most relevant sentences from the retrieved chunks, with inline
  ``[source_file#chunk_index]`` citations. Because it only ever quotes
  sentences that are actually present in the retrieved context, its output
  cannot "hallucinate" facts not present in the source documents — a useful
  property for a financial risk/compliance context where every claim must
  be traceable.

* ``OpenAIAnswerBackend`` (optional) — calls the OpenAI Chat Completions
  API (via the official `openai` python package) using a prompt template
  that instructs the model to answer strictly from the provided context and
  cite sources. Only activated when ``OPENAI_API_KEY`` is set and
  ``LLM_BACKEND=openai``. See ``src/langchain_variant.py`` for the
  equivalent implementation using LangChain's ``PromptTemplate`` +
  ``ChatOpenAI`` chain, which is a near drop-in replacement if this project
  is later migrated onto the full LangChain stack.

``get_llm_backend()`` is the factory used by ``src/rag_pipeline.py`` and
resolves the backend based on ``src.config.settings``.
"""
from __future__ import annotations

import re
from typing import List, Protocol

from src.config import settings
from src.embeddings import _STOPWORDS

# ---------------------------------------------------------------------------
# Prompt template (kept as a plain f-string; a LangChain PromptTemplate
# equivalent lives in src/langchain_variant.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a financial risk and fraud insights assistant for internal bank "
    "analysts. Answer the question using ONLY the provided context excerpts. "
    "Always cite the source document for every claim using the bracketed "
    "citation tags shown in the context (e.g. [source_file.txt#chunk_1]). "
    "If the context does not contain enough information to answer, say so "
    "explicitly rather than guessing."
)

PROMPT_TEMPLATE = """{system_prompt}

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with inline citations):"""


class RetrievedChunkLike(Protocol):
    text: str
    source: str
    chunk_id: str
    chunk_index: int
    score: float


def _format_context(chunks: List) -> str:
    blocks = []
    for c in chunks:
        tag = f"[{c.source}#chunk_{c.chunk_index}]"
        blocks.append(f"{tag}\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: List) -> str:
    """Public helper so callers/tests can inspect exactly what would be sent
    to an LLM, even when running the extractive backend."""
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=_format_context(chunks),
        question=question,
    )


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Short, ALL-CAPS (optionally numbered) section headers (e.g. "2. EXPOSURE
# LIMITS" or "ANALYST NOTES — SETTLEMENT RISK (SYNTHETIC / FICTIONAL)")
# occasionally sit on their own line inside a chunk. Strip them before
# sentence splitting so they don't get glued onto the following sentence in
# the extracted answer.
_HEADER_LINE_RE = re.compile(
    r"(?:^|(?<=[.\n]))\s*\d{0,2}\.?\s*[A-Z][A-Z0-9 /&'()—-]{2,80}\n", re.MULTILINE
)

# Document-metadata header lines (e.g. "Document ID: AN-2024-007",
# "Author: Settlement Risk Analytics Team") that sit at the top of each
# source .txt file, one label per line. Dropped entirely so they never
# pollute the extracted-answer sentences with IDs/team names instead of
# real policy content.
_METADATA_LABELS = ("document id", "effective date", "owner", "distribution", "author")


def _strip_metadata_lines(text: str) -> str:
    kept_lines = [
        line
        for line in text.split("\n")
        if not any(line.strip().lower().startswith(label + ":") for label in _METADATA_LABELS)
    ]
    return "\n".join(kept_lines)


def _split_sentences(text: str) -> List[str]:
    cleaned = _strip_metadata_lines(text)
    cleaned = _HEADER_LINE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = _SENTENCE_SPLIT_RE.split(cleaned)
    return [s.strip() for s in sentences if s.strip()]


def _keyword_overlap_score(question_terms: set, sentence: str) -> tuple:
    sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    overlap = len(question_terms & sentence_terms)
    # Small tie-break toward sentences with more substantive content, but
    # weighted low enough that it only matters when overlap counts are
    # exactly equal — it should never outrank a sentence with more genuine
    # keyword matches.
    length_bonus = min(len(sentence), 200) / 200.0
    return (overlap, length_bonus * 0.01)


class ExtractiveAnswerBackend:
    """
    Zero-dependency, zero-API-key answer generator.

    Strategy: within each retrieved chunk (already ranked by the retriever),
    pick the 1-2 sentences with the highest lexical overlap with the
    question, and present them as a citation-tagged bullet list, ordered by
    the chunk's retrieval score. This guarantees every sentence in the
    answer is a verbatim quote from a real source document.
    """

    name = "extractive"

    def generate(self, question: str, chunks: List) -> str:
        if not chunks:
            return (
                "I could not find any relevant policy or analyst content to "
                "answer this question. Please rephrase or check that the "
                "document index has been built (`python -m src.ingest`)."
            )

        question_terms = set(re.findall(r"[a-z0-9]+", question.lower())) - _STOPWORDS
        lines = []
        for rank, chunk in enumerate(chunks, start=1):
            sentences = _split_sentences(chunk.text)
            if not sentences:
                continue
            scored = sorted(
                sentences,
                key=lambda s: _keyword_overlap_score(question_terms, s),
                reverse=True,
            )
            best = scored[0] if scored[0] else sentences[0]
            # Prefer the top 1 sentence per chunk to keep the answer tight;
            # fall back to the second-best if the top one is very short.
            picked = best
            if len(picked) < 40 and len(scored) > 1:
                picked = picked + " " + scored[1]

            citation = f"[{chunk.source}#chunk_{chunk.chunk_index}]"
            lines.append(
                f"{rank}. {picked.strip()} {citation} (relevance={chunk.score:.3f})"
            )

        header = f'Based on the retrieved policy and analyst documents, here is what is relevant to: "{question}"\n'
        return header + "\n".join(lines)


class OpenAIAnswerBackend:
    """
    Calls OpenAI's Chat Completions API to synthesize a natural-language,
    citation-grounded answer from the retrieved context.

    Only used when LLM_BACKEND=openai and OPENAI_API_KEY is set; otherwise
    src.rag_pipeline falls back to ExtractiveAnswerBackend automatically
    (see Settings.effective_llm_backend()).
    """

    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The `openai` package is not installed. Install it with "
                "`pip install openai`, or unset OPENAI_API_KEY / set "
                "LLM_BACKEND=extractive to use the offline backend."
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate(self, question: str, chunks: List) -> str:
        if not chunks:
            return (
                "I could not find any relevant policy or analyst content to "
                "answer this question based on the indexed documents."
            )
        prompt = build_prompt(question, chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content


def get_llm_backend():
    """Factory resolving the configured backend, with a safe fallback to
    the extractive backend if OpenAI is requested but unavailable."""
    backend_name = settings.effective_llm_backend()
    if backend_name == "openai":
        try:
            return OpenAIAnswerBackend(
                api_key=settings.openai_api_key, model=settings.openai_model
            )
        except ImportError:
            return ExtractiveAnswerBackend()
    return ExtractiveAnswerBackend()
