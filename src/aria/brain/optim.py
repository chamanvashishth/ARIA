"""Small local optimizers for ARIA model parameters."""

from __future__ import annotations

from aria.brain.parameter import Parameter


class SGD:
    """Plain stochastic gradient descent."""

    def __init__(self, parameters: list[Parameter], learning_rate: float = 0.01) -> None:
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
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


def _reshape(values: list[float], shape: tuple[int, ...]) -> object:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values)
    width = 1
    for dim in shape[1:]:
        width *= dim
    return [_reshape(values[i * width:(i + 1) * width], shape[1:]) for i in range(shape[0])]
