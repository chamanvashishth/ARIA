"""Trainable neural layers built on ARIA's tensor foundation."""

from __future__ import annotations

import math
import random
from typing import Sequence

from aria.brain.module import Module
from aria.brain.parameter import Parameter
from aria.brain.tensor import Tensor


class Linear(Module):
    """A fully connected layer y = xW + b with gradient propagation."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, seed: int | None = None) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("layer dimensions must be positive")
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / in_features)
        self.weight = Parameter([[rng.uniform(-scale, scale) for _ in range(out_features)] for _ in range(in_features)])
        self.bias = Parameter([0.0] * out_features) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) < 2:
            raise ValueError("Linear input must have at least 2 dimensions")
        leading_shape = x.shape[:-1]
        rows = math.prod(leading_shape)
        in_features = x.shape[-1]
        if in_features != self.weight.shape[0]:
            raise ValueError("Linear input dimension does not match weight")
        out_features = self.weight.shape[1]
        xv, wv = x._values, self.weight._values
        values = [
            sum(xv[row * in_features + k] * wv[k * out_features + col] for k in range(in_features))
            + (self.bias._values[col] if self.bias is not None else 0.0)
            for row in range(rows)
            for col in range(out_features)
        ]

        def backward(out: Tensor) -> None:
            grad = out.grad._values
            if x.requires_grad:
                dx = [
                    sum(grad[row * out_features + col] * wv[k * out_features + col] for col in range(out_features))
                    for row in range(rows)
                    for k in range(in_features)
                ]
                x._accumulate(dx)
            if self.weight.requires_grad:
                dw = [
                    sum(xv[row * in_features + k] * grad[row * out_features + col] for row in range(rows))
                    for k in range(in_features)
                    for col in range(out_features)
                ]
                self.weight._accumulate(dw)
            if self.bias is not None and self.bias.requires_grad:
                self.bias._accumulate([
                    sum(grad[row * out_features + col] for row in range(rows))
                    for col in range(out_features)
                ])

        parents = (x, self.weight, *((self.bias,) if self.bias is not None else ()))
        return Tensor.operation(_reshape(values, (*leading_shape, out_features)), parents=parents, backward=backward)



class ReLU(Module):
    """Rectified linear activation with gradient propagation."""

    def forward(self, x: Tensor) -> Tensor:
        values = [max(0.0, value) for value in x._values]

        def backward(out: Tensor) -> None:
            if x.requires_grad:
                x._accumulate([g if value > 0.0 else 0.0 for g, value in zip(out.grad._values, x._values)])

        return Tensor.operation(_reshape(values, x.shape), parents=(x,), backward=backward)


def _reshape(values: Sequence[float], shape: tuple[int, ...]) -> object:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values)
    width = math.prod(shape[1:])
    return [_reshape(values[i * width:(i + 1) * width], shape[1:]) for i in range(shape[0])]


class Embedding(Module):
    """Trainable lookup table for integer token IDs."""

    def __init__(self, num_embeddings: int, embedding_dim: int, *, seed: int | None = None) -> None:
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(embedding_dim)
        self.weight = Parameter([[rng.uniform(-scale, scale) for _ in range(embedding_dim)] for _ in range(num_embeddings)])

    def forward(self, token_ids: Sequence[int]) -> Tensor:
        rows = []
        for token_id in token_ids:
            if not 0 <= token_id < self.weight.shape[0]:
                raise IndexError(f"token id out of range: {token_id}")
            start = token_id * self.weight.shape[1]
            rows.append(self.weight._values[start:start + self.weight.shape[1]])

        embedding_dim = self.weight.shape[1]

        def backward(out: Tensor) -> None:
            if not self.weight.requires_grad:
                return
            grad = [0.0] * len(self.weight._values)
            for row, token_id in enumerate(token_ids):
                start = token_id * embedding_dim
                for col in range(embedding_dim):
                    grad[start + col] += out.grad._values[row * embedding_dim + col]
            self.weight._accumulate(grad)

        return Tensor.operation(rows, parents=(self.weight,), backward=backward)


class Sequential(Module):
    """Apply a sequence of modules in order."""

    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x
