"""Trainable parameter primitive."""

from aria.brain.tensor import Tensor


class Parameter(Tensor):
    """A tensor representing a trainable model parameter."""

    def __init__(self, data: object) -> None:
        super().__init__(data, requires_grad=True)
