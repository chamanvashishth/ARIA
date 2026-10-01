"""Data structures for local RAG documents and deterministic chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class TextDocument:
    """A plain-text document with a caller-supplied source identifier."""

    source: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of source text with stable identity and provenance metadata."""

    id: str
    source: str
    text: str
    chunk_index: int
    metadata: Mapping[str, str] = field(default_factory=dict)
