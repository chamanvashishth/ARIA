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

    def forward_batch(self, token_batches: list[list[int]]) -> Tensor:
        """Project a fixed-length batch through one embedding and one Linear call.

        The result has shape [batch * time, vocab_size], with rows ordered by
        batch first and then time. This model has no attention, so flattening
        the token positions does not create cross-example interactions.
        """
        if not token_batches:
            raise ValueError("batch must contain at least one sequence")
        sequence_length = len(token_batches[0])
        if sequence_length == 0:
            raise ValueError("token sequences cannot be empty")
        if any(len(sequence) != sequence_length for sequence in token_batches):
            raise ValueError("all batch sequences must have the same length")
        flattened_ids = [token for sequence in token_batches for token in sequence]
        hidden = self.embedding.forward(flattened_ids)
        return self.vocabulary.forward(hidden)

    def loss(self, token_ids: list[int], targets: list[int]) -> Tensor:
        return _log_softmax_loss(self.forward(token_ids), targets)

    def loss_batch(
        self,
        token_batches: list[list[int]],
        target_batches: list[list[int]],
    ) -> Tensor:
        """Compute mean next-token loss across an equal-length batch."""
        if len(token_batches) != len(target_batches) or not token_batches:
            raise ValueError("input and target batches must have equal non-zero size")
        if any(len(inputs) != len(targets) for inputs, targets in zip(token_batches, target_batches)):
            raise ValueError("each input sequence must match its target length")
        flat_targets = [target for sequence in target_batches for target in sequence]
        return _log_softmax_loss(self.forward_batch(token_batches), flat_targets)
