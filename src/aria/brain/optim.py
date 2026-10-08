"""Small local optimizers for ARIA model parameters."""

from __future__ import annotations

import math

from aria.brain.parameter import Parameter


class SGD:
    """Plain stochastic gradient descent."""

    def __init__(self, parameters: list[Parameter], learning_rate: float = 0.01) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            parameter._values = [
                value - self.learning_rate * gradient
                for value, gradient in zip(parameter._values, parameter.grad._values)
            ]
            parameter.data = _reshape(parameter._values, parameter.shape)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class LearningRateScheduler:
    """Base interface for deterministic optimizer learning-rate schedules."""

    def step(self, optimizer: SGD, completed_steps: int) -> float:
        raise NotImplementedError


class ExponentialDecay(LearningRateScheduler):
    """Multiply the learning rate by gamma after each completed step."""

    def __init__(self, gamma: float = 0.99, *, minimum: float = 0.0) -> None:
        if not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must be finite and between 0 and 1")
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError("minimum learning rate must be finite and non-negative")
        self.gamma = gamma
        self.minimum = minimum

    def step(self, optimizer: SGD, completed_steps: int) -> float:
        if completed_steps <= 0:
            raise ValueError("completed_steps must be positive")
        optimizer.learning_rate = max(
            self.minimum,
            optimizer.learning_rate * self.gamma,
        )
        return optimizer.learning_rate


class StepDecay(LearningRateScheduler):
    """Drop the learning rate by gamma every fixed number of steps."""

    def __init__(
        self,
        drop_every: int,
        gamma: float = 0.5,
        *,
        minimum: float = 0.0,
    ) -> None:
        if drop_every <= 0:
            raise ValueError("drop_every must be positive")
        if not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must be finite and between 0 and 1")
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError("minimum learning rate must be finite and non-negative")
        self.drop_every = drop_every
        self.gamma = gamma
        self.minimum = minimum

    def step(self, optimizer: SGD, completed_steps: int) -> float:
        if completed_steps <= 0:
            raise ValueError("completed_steps must be positive")
        if completed_steps % self.drop_every == 0:
            optimizer.learning_rate = max(
                self.minimum,
                optimizer.learning_rate * self.gamma,
            )
        return optimizer.learning_rate


def _reshape(values: list[float], shape: tuple[int, ...]) -> object:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values)
    width = 1
    for dim in shape[1:]:
        width *= dim
    return [_reshape(values[i * width:(i + 1) * width], shape[1:]) for i in range(shape[0])]
