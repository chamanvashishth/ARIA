"""Trainable neural layers built on ARIA's tensor foundation."""

from __future__ import annotations

import math
import random
from typing import Sequence

from aria.brain.module import Module
from aria.brain.parameter import Parameter
from aria.brain.tensor import Tensor


def _matmul_2d(x: Tensor, weight: Parameter) -> Tensor:
    if len(x.shape) != 2 or len(weight.shape) != 2:
        raise ValueError("Linear currently requires 2-D tensors")
    batch, in_features = x.shape
    weight_in, out_features = weight.shape
    if in_features != weight_in:
        raise ValueError("Linear input dimension does not match weight")

    x_values = x._values
    w_values = weight._values
    rows = []
    for row in range(batch):
        for col in range(out_features):
            rows.append(
                sum(
                    x_values[row * in_features + k] * w_values[k * out_features + col]
                    for k in range(in_features)
                )
            )
    return Tensor.from_list(
        [rows[i * out_features:(i + 1) * out_features] for i in range(batch)],
        requires_grad=x.requires_grad or weight.requires_grad,
    )


class Linear(Module):
    """A fully connected layer y = xW + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("layer dimensions must be positive")

        rng = random.Random(seed)
        scale = math.sqrt(2.0 / in_features)
        self.weight = Parameter(
            [
                [rng.uniform(-scale, scale) for _ in range(out_features)]
                for _ in range(in_features)
            ]
        )
        self.bias = Parameter([0.0] * out_features) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        output = _matmul_2d(x, self.weight)
        if self.bias is None:
            return output

        bias_values = self.bias._values
        values = [
            value + bias_values[col]
            for row in range(x.shape[0])
            for col, value in enumerate(output._values[row * output.shape[1]:(row + 1) * output.shape[1]])
        ]
        return Tensor.from_list(
            [values[i * output.shape[1]:(i + 1) * output.shape[1]] for i in range(x.shape[0])],
            requires_grad=output.requires_grad or self.bias.requires_grad,
        )


class ReLU(Module):
    """Rectified linear activation."""

    def forward(self, x: Tensor) -> Tensor:
        values = [max(0.0, value) for value in x._values]
        return Tensor(_build_like(values, x), requires_grad=x.requires_grad, _parents=(x,))

def _build_like(values: Sequence[float], reference: Tensor) -> object:
    if not reference.shape:
        return float(values[0])
    if len(reference.shape) == 1:
        return list(values)
    width = math.prod(reference.shape[1:])
    return [
        _build_like(values[i * width:(i + 1) * width], Tensor.zeros(reference.shape[1:]))
        for i in range(reference.shape[0])
    ]


class Embedding(Module):
    """Trainable lookup table for integer token IDs."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        seed: int | None = None,
    ) -> None:
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(embedding_dim)
        self.weight = Parameter(
            [
                [rng.uniform(-scale, scale) for _ in range(embedding_dim)]
                for _ in range(num_embeddings)
            ]
        )

    def forward(self, token_ids: Sequence[int]) -> Tensor:
        rows = []
        for token_id in token_ids:
            if not 0 <= token_id < self.weight.shape[0]:
                raise IndexError(f"token id out of range: {token_id}")
            start = token_id * self.weight.shape[1]
            end = start + self.weight.shape[1]
            rows.append(self.weight._values[start:end])
        return Tensor.from_list(rows, requires_grad=self.weight.requires_grad)


class Sequential(Module):
    """Apply a sequence of modules in order."""

    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x
