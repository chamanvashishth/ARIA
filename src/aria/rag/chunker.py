"""Deterministic character-window chunking for local text documents."""

from __future__ import annotations

import hashlib

from aria.rag.documents import DocumentChunk, TextDocument


class TextChunker:
    """Split documents into overlapping character windows.

    Chunking is deliberately format-agnostic: callers provide extracted text.
    Source metadata is copied to every chunk so retrieval results retain
    provenance. Chunk IDs are stable for identical source/index/content input.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0:
            raise ValueError("overlap must not be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: TextDocument) -> list[DocumentChunk]:
        if not document.text:
            return []

        chunks: list[DocumentChunk] = []
        step = self.chunk_size - self.overlap
        for index, start in enumerate(range(0, len(document.text), step)):
            # If the remaining suffix fits entirely inside the previous
            # chunk's overlap, it adds no new source text and is redundant.
            if len(document.text) - start <= self.overlap:
                break
            text = document.text[start : start + self.chunk_size]
            if not text:
                continue
            digest = hashlib.sha256(
                f"{document.source}\0{index}\0{text}".encode("utf-8")
            ).hexdigest()[:20]
            chunks.append(
                DocumentChunk(
                    id=digest,
                    source=document.source,
                    text=text,
                    chunk_index=index,
                    metadata=dict(document.metadata),
                )
            )
        return chunks
