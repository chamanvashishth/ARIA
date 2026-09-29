"""Validation helpers for SLM training examples."""

from collections.abc import Iterable

from .config import SLMConfig
from .dataset import TrainingExample


def validate_examples(examples: Iterable[TrainingExample], config: SLMConfig) -> int:
    """Validate training examples against the model configuration.

    Returns the number of validated examples. Token IDs must fit inside the
    configured vocabulary and each example must match the context length.
    """
    count = 0
    for index, example in enumerate(examples):
        if not isinstance(example, TrainingExample):
            raise TypeError(f"example {index} must be a TrainingExample")
        if len(example.input_ids) != config.context_length:
            raise ValueError(
                f"example {index} input length must equal context_length ({config.context_length})"
            )
        if any(token >= config.vocab_size for token in (*example.input_ids, *example.target_ids)):
            raise ValueError(f"example {index} contains an ID outside vocab_size")
        count += 1
    return count
