"""Tokenizer contract for the ARIA SLM."""

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Convert text to token IDs and token IDs back to text.

    Implementations must return deterministic integer token IDs from encode
    and accept integer token IDs in decode. The tokenizer owns
    vocabulary-specific rules; the SLM only depends on this boundary.
    """

    def encode(self, text: str) -> Sequence[int]:
        """Encode text into an ordered sequence of token IDs."""
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode an ordered sequence of token IDs into text."""
        ...
