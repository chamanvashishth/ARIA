"""Small trainable autoregressive language-model core."""

from __future__ import annotations

import math

from aria.brain.layers import Embedding, Linear
from aria.brain.module import Module
from aria.brain.tensor import Tensor


def _log_softmax_loss(logits: Tensor, targets: list[int]) -> Tensor:
    """Compute mean next-token cross-entropy for a [time, vocab] matrix."""
    if len(logits.shape) != 2:
        raise ValueError("logits must be a 2-D tensor")
    time, vocab = logits.shape
    if len(targets) != time:
        raise ValueError("target count must match logits time dimension")
    if not targets:
        raise ValueError("at least one target is required")

    values = logits._values
    losses = []
    for row, target in enumerate(targets):
        if not 0 <= target < vocab:
            raise IndexError(f"target token out of range: {target}")
        start = row * vocab
        row_values = values[start:start + vocab]
        maximum = max(row_values)
        log_sum_exp = maximum + math.log(sum(math.exp(v - maximum) for v in row_values))
        losses.append(log_sum_exp - row_values[target])

    mean_loss = sum(losses) / time

    def backward(out: Tensor) -> None:
        grad = [0.0] * len(values)
        scale = out.grad._values[0] / time
        for row, target in enumerate(targets):
            start = row * vocab
            row_values = values[start:start + vocab]
            maximum = max(row_values)
            exp_values = [math.exp(v - maximum) for v in row_values]
            normalizer = sum(exp_values)
            for col, exp_value in enumerate(exp_values):
                grad[start + col] = scale * (exp_value / normalizer - (1.0 if col == target else 0.0))
        if logits.requires_grad:
            logits._accumulate(grad)

    return Tensor.operation(mean_loss, parents=(logits,), backward=backward)


class TinyLanguageModel(Module):
    """A small next-token model used to validate ARIA's learning pipeline.

    This is a foundation model, not the final Transformer SLM.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        seed: int | None = None,
    ) -> None:
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("model dimensions must be positive")
        self.embedding = Embedding(vocab_size, embedding_dim, seed=seed)
        self.vocabulary = Linear(embedding_dim, vocab_size, seed=seed)

    def forward(self, token_ids: list[int]) -> Tensor:
        hidden = self.embedding.forward(token_ids)
        return self.vocabulary.forward(hidden)

    def loss(self, token_ids: list[int], targets: list[int]) -> Tensor:
        return _log_softmax_loss(self.forward(token_ids), targets)
