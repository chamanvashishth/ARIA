"""Small, deterministic software execution engines.

These engines are ARIA components. They do not require physical accelerator
hardware. Host acceleration can be added later without changing their API.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T", int, float)


@dataclass(frozen=True)
class VirtualCPU:
    """General-purpose scalar and vector execution."""

    name: str = "ARIA Virtual CPU"

    def map(self, values: Sequence[T], operation) -> list[T]:
        return [operation(value) for value in values]


@dataclass(frozen=True)
class VirtualGPU:
    """Parallel-style matrix execution implemented in software."""

    name: str = "ARIA Virtual GPU"

    def matmul(
        self,
        left: Sequence[Sequence[T]],
        right: Sequence[Sequence[T]],
    ) -> list[list[T]]:
        if not left or not right:
            raise ValueError("matrices must not be empty")

        left_width = len(left[0])
        if left_width == 0 or any(len(row) != left_width for row in left):
            raise ValueError("left matrix must be rectangular")

        right_width = len(right[0])
        if right_width == 0 or any(len(row) != right_width for row in right):
            raise ValueError("right matrix must be rectangular")

        if len(right) != left_width:
            raise ValueError("matrix dimensions do not align")

        return [
            [
                sum(left[row][k] * right[k][column] for k in range(left_width))
                for column in range(right_width)
            ]
            for row in range(len(left))
        ]


@dataclass(frozen=True)
class VirtualNPU:
    """Neural-network-oriented matrix execution implemented in software."""

    name: str = "ARIA Virtual NPU"

    def dense(
        self,
        inputs: Sequence[Sequence[float]],
        weights: Sequence[Sequence[float]],
        bias: Sequence[float] | None = None,
    ) -> list[list[float]]:
        result = VirtualGPU().matmul(inputs, weights)

        if bias is None:
            return result

        if len(result[0]) != len(bias):
            raise ValueError("bias width does not match dense output")

        return [
            [value + bias[column] for column, value in enumerate(row)]
            for row in result
        ]

    def relu(self, values: Sequence[float]) -> list[float]:
        return [max(0.0, value) for value in values]
