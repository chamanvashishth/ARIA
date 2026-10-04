"""Dependency-free evaluation metrics for local ARIA language models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from aria.brain.language_model import _log_softmax_loss
from aria.brain.module import Module


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate next-token metrics for a collection of input/target examples."""

    mean_loss: float
    perplexity: float
    token_accuracy: float
    token_count: int
    batch_count: int


@dataclass(frozen=True)
class ParameterHealth:
    """Finite-value audit for model parameters and any accumulated gradients."""

    parameter_count: int
    non_finite_values: int
    non_finite_gradients: int

    @property
    def healthy(self) -> bool:
        return self.non_finite_values == 0 and self.non_finite_gradients == 0


def _perplexity(mean_loss: float) -> float:
    """Convert mean cross-entropy to perplexity without overflowing."""
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return math.inf


def evaluate_language_model(
    model: Module,
    examples: Iterable[tuple[list[int], list[int]]],
) -> EvaluationResult:
    """Measure mean next-token loss, perplexity, and exact token accuracy.

    Each example contains input token IDs and the expected next token at each
    position. Evaluation performs forward passes only: it does not call
    backward(), update parameters, or clear existing gradients.
    """
    weighted_loss = 0.0
    token_count = 0
    correct_tokens = 0
    batch_count = 0

    for token_ids, targets in examples:
        if not token_ids:
            raise ValueError("evaluation input cannot be empty")
        if not targets:
            raise ValueError("evaluation targets cannot be empty")

        logits = model.forward(token_ids)
        if len(logits.shape) != 2 or logits.shape[0] != len(targets):
            raise ValueError("model logits must have one row per target token")
        if not all(math.isfinite(value) for value in logits._values):
            raise ValueError("model produced non-finite logits")

        loss = _log_softmax_loss(logits, targets).item()
        if not math.isfinite(loss):
            raise ValueError("model produced a non-finite evaluation loss")

        vocab_size = logits.shape[1]
        values = logits._values
        for row, target in enumerate(targets):
            if not 0 <= target < vocab_size:
                raise IndexError(f"target token out of range: {target}")
            start = row * vocab_size
            row_values = values[start:start + vocab_size]
            prediction = max(range(vocab_size), key=row_values.__getitem__)
            correct_tokens += int(prediction == target)

        count = len(targets)
        weighted_loss += loss * count
        token_count += count
        batch_count += 1

    if token_count == 0:
        raise ValueError("evaluation requires at least one example")

    mean_loss = weighted_loss / token_count
    return EvaluationResult(
        mean_loss=mean_loss,
        perplexity=_perplexity(mean_loss),
        token_accuracy=correct_tokens / token_count,
        token_count=token_count,
        batch_count=batch_count,
    )


def inspect_parameter_health(model: Module) -> ParameterHealth:
    """Count non-finite parameter values and accumulated gradient values."""
    parameters = model.parameters()
    parameter_count = 0
    non_finite_values = 0
    non_finite_gradients = 0

    for parameter in parameters:
        parameter_count += len(parameter._values)
        non_finite_values += sum(not math.isfinite(value) for value in parameter._values)
        if parameter.grad is not None:
            non_finite_gradients += sum(
                not math.isfinite(value) for value in parameter.grad._values
            )

    return ParameterHealth(
        parameter_count=parameter_count,
        non_finite_values=non_finite_values,
        non_finite_gradients=non_finite_gradients,
    )
