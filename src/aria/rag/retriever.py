"""Dependency-free lexical retrieval using BM25-style term scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from aria.rag.documents import DocumentChunk


_TOKEN_PATTERN = re.compile(r"(?:[^\W_]|[\u0900-\u097F])+", flags=re.UNICODE)


def tokenize_terms(text: str) -> list[str]:
    """Return normalized Unicode word terms for lexical matching."""
    return [match.casefold() for match in _TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    """Rank chunks by lexical overlap; this is not semantic retrieval."""

    def __init__(
        self,
        chunks: Iterable[DocumentChunk] = (),
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.k1 = k1
        self.b = b
        self._chunks: list[DocumentChunk] = []
        self._terms: list[list[str]] = []
        self._term_counts: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0
        self.replace_chunks(chunks)

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        return tuple(self._chunks)

    def replace_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._terms = [tokenize_terms(chunk.text) for chunk in self._chunks]
        self._term_counts = [Counter(terms) for terms in self._terms]
        self._document_frequency = Counter()
        for counts in self._term_counts:
            self._document_frequency.update(counts.keys())
        total_terms = sum(len(terms) for terms in self._terms)
        self._average_length = total_terms / len(self._terms) if self._terms else 0.0

    def add_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        self.replace_chunks([*self._chunks, *chunks])

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if top_k == 0 or not self._chunks:
            return []

        query_terms = set(tokenize_terms(query))
        if not query_terms:
            return []

        count = len(self._chunks)
        scored: list[tuple[int, DocumentChunk, float]] = []
        for index, (chunk, terms, counts) in enumerate(
            zip(self._chunks, self._terms, self._term_counts)
        ):
            length = len(terms)
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / (self._average_length or 1.0)
                )
                score += inverse_frequency * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            if score > 0:
                scored.append((index, chunk, score))

        scored.sort(key=lambda item: (-item[2], item[0]))
        return [(chunk, score) for _, chunk, score in scored[:top_k]]
