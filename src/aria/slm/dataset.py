"""Dataset contracts for SLM training data."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One language-model training pair of input and target token IDs."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.input_ids or not self.target_ids:
            raise ValueError("input_ids and target_ids must not be empty")
        if len(self.input_ids) != len(self.target_ids):
            raise ValueError("input_ids and target_ids must have the same length")
        for name, tokens in (("input_ids", self.input_ids), ("target_ids", self.target_ids)):
            if any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
                raise TypeError(f"{name} must contain only integers")
            if any(token < 0 for token in tokens):
                raise ValueError(f"{name} must contain only non-negative IDs")


@runtime_checkable
class Dataset(Protocol):
    """Minimal indexed dataset boundary consumed by the SLM trainer."""

    def __len__(self) -> int:
        """Return the number of training examples."""
        ...

    def __getitem__(self, index: int) -> TrainingExample:
        """Return the training example at an integer index."""
        ...
