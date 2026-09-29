"""Deterministic preprocessing from text into SLM training examples."""

from .dataset import TrainingExample
from .tokenizer import Tokenizer


def preprocess_text(text: str, tokenizer: Tokenizer, context_length: int) -> tuple[TrainingExample, ...]:
    """Tokenize text and create fixed next-token training windows."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        raise TypeError("context_length must be an integer")
    if context_length <= 0:
        raise ValueError("context_length must be greater than zero")

    tokens = tuple(tokenizer.encode(text))
    if any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
        raise TypeError("tokenizer.encode must return only integers")
    if any(token < 0 for token in tokens):
        raise ValueError("tokenizer.encode must return only non-negative IDs")

    window_size = context_length + 1
    return tuple(
        TrainingExample(
            input_ids=tokens[start : start + context_length],
            target_ids=tokens[start + 1 : start + window_size],
        )
        for start in range(0, len(tokens) - context_length, context_length)
        if len(tokens[start : start + window_size]) == window_size
    )
