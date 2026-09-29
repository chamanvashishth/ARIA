"""Validated configuration for an ARIA Small Language Model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SLMConfig:
    """Describe the architectural dimensions of an SLM instance.

    The configuration is immutable so a model cannot silently change shape
    after construction. Validation happens at the boundary before any model
    implementation consumes these values.
    """

    vocab_size: int
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a number")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
