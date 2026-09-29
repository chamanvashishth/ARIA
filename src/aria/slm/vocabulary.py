"""Immutable vocabulary mapping for the ARIA SLM."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Map unique token strings to stable zero-based integer IDs."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("tokens must not be empty")
        if any(not isinstance(token, str) or not token for token in self.tokens):
            raise ValueError("tokens must be non-empty strings")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("tokens must be unique")

    @property
    def size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.tokens)

    def token_to_id(self, token: str) -> int:
        """Return the stable ID assigned to a token."""
        try:
            return self.tokens.index(token)
        except ValueError as exc:
            raise KeyError(token) from exc

    def id_to_token(self, token_id: int) -> str:
        """Return the token assigned to a stable integer ID."""
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token_id must be an integer")
        try:
            return self.tokens[token_id]
        except IndexError as exc:
            raise KeyError(token_id) from exc
