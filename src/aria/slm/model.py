"""Model architecture contract for the ARIA SLM."""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Logits produced for each token position in a batch."""

    logits: tuple[tuple[tuple[float, ...], ...], ...]

    @property
    def batch_size(self) -> int:
        """Return the number of sequences in the output batch."""
        return len(self.logits)

    @property
    def sequence_length(self) -> int:
        """Return the number of token positions per sequence."""
        return len(self.logits[0]) if self.logits else 0

    @property
    def vocab_size(self) -> int:
        """Return the number of logits per token position."""
        if not self.logits or not self.logits[0]:
            return 0
        return len(self.logits[0][0])


@runtime_checkable
class SLMModel(Protocol):
    """Minimal forward-pass boundary for a language model."""

    def forward(self, input_ids: Sequence[Sequence[int]]) -> ModelOutput:
        """Produce vocabulary logits for every input token position."""
        ...
