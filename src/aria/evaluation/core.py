"""Dependency-free evaluation and numerical validation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

from aria.brain.language_model import _log_softmax_loss
from aria.brain.module import Module
from aria.brain.parameter import Parameter
from aria.brain.tensor import Tensor, _build


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


@dataclass(frozen=True)
class GradientCheckResult:
    """Comparison between reverse-mode gradients and finite differences."""

    checked_values: int
    mismatched_values: int
    max_absolute_error: float
    max_relative_error: float

    @property
    def passed(self) -> bool:
        return self.checked_values > 0 and self.mismatched_values == 0


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


def check_gradients(
    loss_fn: Callable[[], Tensor],
    parameters: Iterable[Parameter],
    *,
    epsilon: float = 1e-5,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-3,
    max_checks: int | None = 32,
) -> GradientCheckResult:
    """Compare analytical gradients with central finite differences.

    loss_fn must deterministically return a scalar Tensor and should not update
    parameters. Existing parameter gradients are cleared, and the analytical
    gradients from the check remain attached afterward. For large models,
    max_checks bounds work to the first values in parameter order; pass None
    to inspect every value. This is a diagnostic, not a proof that every
    operation or model path is correct.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be finite and non-negative")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    if max_checks is not None and max_checks <= 0:
        raise ValueError("max_checks must be positive or None")

    parameter_list = list(parameters)
    if not parameter_list:
        raise ValueError("at least one parameter is required")

    selected: list[tuple[Parameter, int]] = []
    for parameter in parameter_list:
        for index in range(len(parameter._values)):
            selected.append((parameter, index))
            if max_checks is not None and len(selected) >= max_checks:
                break
        if max_checks is not None and len(selected) >= max_checks:
            break
    if not selected:
        raise ValueError("parameters contain no values")

    original_values = {id(p): list(p._values) for p in parameter_list}
    try:
        for parameter in parameter_list:
            parameter.zero_grad()

        analytical_loss = loss_fn()
        if analytical_loss.shape != () or not analytical_loss.requires_grad:
            raise ValueError("loss_fn must return a scalar tensor that requires gradients")
        if not math.isfinite(analytical_loss.item()):
            raise ValueError("loss_fn returned a non-finite loss")
        analytical_loss.backward()

        analytical: list[float] = []
        for parameter, index in selected:
            if parameter.grad is None:
                analytical.append(0.0)
            elif len(parameter.grad._values) != len(parameter._values):
                raise ValueError("parameter gradient shape does not match parameter")
            else:
                analytical.append(parameter.grad._values[index])

        numerical: list[float] = []
        for parameter, index in selected:
            base = original_values[id(parameter)][index]
            parameter._values[index] = base + epsilon
            parameter.data = _build(parameter._values, parameter.shape)
            plus_loss = loss_fn()
            if plus_loss.shape != () or not math.isfinite(plus_loss.item()):
                raise ValueError("loss_fn returned an invalid loss during finite differences")

            parameter._values[index] = base - epsilon
            parameter.data = _build(parameter._values, parameter.shape)
            minus_loss = loss_fn()
            if minus_loss.shape != () or not math.isfinite(minus_loss.item()):
                raise ValueError("loss_fn returned an invalid loss during finite differences")

            numerical.append((plus_loss.item() - minus_loss.item()) / (2.0 * epsilon))
            parameter._values[index] = base
            parameter.data = _build(parameter._values, parameter.shape)

        mismatches = 0
        max_absolute_error = 0.0
        max_relative_error = 0.0
        for analytic_value, numeric_value in zip(analytical, numerical):
            absolute_error = abs(analytic_value - numeric_value)
            relative_error = absolute_error / max(
                abs(analytic_value), abs(numeric_value), 1e-12
            )
            max_absolute_error = max(max_absolute_error, absolute_error)
            max_relative_error = max(max_relative_error, relative_error)
            if (
                not math.isfinite(analytic_value)
                or not math.isfinite(numeric_value)
                or absolute_error
                > absolute_tolerance
                + relative_tolerance * max(abs(analytic_value), abs(numeric_value))
            ):
                mismatches += 1

        return GradientCheckResult(
            checked_values=len(selected),
            mismatched_values=mismatches,
            max_absolute_error=max_absolute_error,
            max_relative_error=max_relative_error,
        )
    finally:
        for parameter in parameter_list:
            parameter._values = original_values[id(parameter)]
            parameter.data = _build(parameter._values, parameter.shape)
