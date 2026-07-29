"""
Document chunking.

``split_text`` implements a recursive character-based splitter: it tries to
break on paragraph boundaries first, then sentence boundaries, then falls
back to a hard character-window split with overlap. This is functionally
equivalent to LangChain's ``RecursiveCharacterTextSplitter`` with
separators=["\\n\\n", "\\n", ". ", " "], which is why the project can adopt
LangChain later with only an import swap — see ``src/langchain_variant.py``
for the literal drop-in version.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    chunk_id: str
    source: str
    chunk_index: int


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Recursively split `text` using the first separator that yields pieces
    small enough to fit chunk_size; falls back to a hard split at the end."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Hard split as last resort.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, *rest = separators
    pieces = _split_on_separator(text, sep)

    # Re-glue the separator (except for the space/hard split cases where it's fine to drop)
    joiner = sep if sep not in ("", " ") else sep

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + joiner + piece) if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(piece) > chunk_size:
                # piece itself is too big; recurse with the next-level separators
                chunks.extend(_recursive_split(piece, chunk_size, rest))
                current = ""
            else:
                current = piece
    if current.strip():
        chunks.append(current)

    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + chunks[i])
    return overlapped


def split_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> List[str]:
    """Split `text` into overlapping chunks, respecting natural boundaries
    (paragraphs > lines > sentences > words) where possible."""
    text = text.strip()
    if not text:
        return []
    raw_chunks = _recursive_split(text, chunk_size, _SEPARATORS)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    return _add_overlap(raw_chunks, chunk_overlap)


def chunk_document(text: str, source: str, chunk_size: int = 700, chunk_overlap: int = 120) -> List[Chunk]:
    """Chunk a single document's text and attach source metadata."""
    pieces = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        Chunk(
            text=piece,
            chunk_id=f"{source}::chunk_{i}",
            source=source,
            chunk_index=i,
        )
        for i, piece in enumerate(pieces)
    ]
