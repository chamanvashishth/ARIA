"""Local retrieval pipeline and prompt-context assembly."""

from __future__ import annotations

from collections.abc import Iterable

from aria.rag.chunker import TextChunker
from aria.rag.documents import DocumentChunk, TextDocument
from aria.rag.retriever import BM25Retriever


class LocalRAGPipeline:
    """Ingest plain text and retrieve ranked lexical context locally.

    This class assembles context only; it does not generate answers, inspect
    PDFs/websites, or guarantee that a downstream model will follow the context.
    """

    def __init__(
        self,
        *,
        chunker: TextChunker | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        self.chunker = chunker or TextChunker()
        self.top_k = top_k
        self._chunks: list[DocumentChunk] = []
        self.retriever = BM25Retriever()

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        return tuple(self._chunks)

    def add_document(self, document: TextDocument) -> list[DocumentChunk]:
        new_chunks = self.chunker.chunk(document)
        self._chunks.extend(new_chunks)
        self.retriever.replace_chunks(self._chunks)
        return new_chunks

    def add_documents(self, documents: Iterable[TextDocument]) -> list[DocumentChunk]:
        added: list[DocumentChunk] = []
        for document in documents:
            added.extend(self.chunker.chunk(document))
        if added:
            self._chunks.extend(added)
            self.retriever.replace_chunks(self._chunks)
        return added

    def retrieve(
        self, query: str, *, top_k: int | None = None
    ) -> list[tuple[DocumentChunk, float]]:
        return self.retriever.search(
            query, top_k=self.top_k if top_k is None else top_k
        )

    def build_context(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_characters: int = 6000,
    ) -> str:
        """Format retrieved excerpts with source labels within a character cap."""
        if max_characters < 0:
            raise ValueError("max_characters must not be negative")
        if max_characters == 0:
            return ""

        sections: list[str] = []
        used = 0
        for chunk, _score in self.retrieve(query, top_k=top_k):
            section = f"[Source: {chunk.source} | Chunk: {chunk.chunk_index}]\n{chunk.text}"
            separator = "\n\n" if sections else ""
            remaining = max_characters - used - len(separator)
            if remaining <= 0:
                break
            if len(section) > remaining:
                section = section[:remaining]
            sections.append(section)
            used += len(separator) + len(section)
            if used >= max_characters:
                break
        return "\n\n".join(sections)
