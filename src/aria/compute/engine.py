"""ARIA-owned software execution engines."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T", int, float)


def _matrix_shape(matrix: Sequence[Sequence[T]], name: str) -> tuple[int, int]:
    if not matrix:
        raise ValueError(f"{name} must not be empty")

    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")

    return len(matrix), width


@dataclass(frozen=True, slots=True)
class VirtualCPU:
    """General-purpose software execution."""

    name: str = "ARIA Virtual CPU"

    def map(self, values: Sequence[T], operation) -> list[T]:
        if not callable(operation):
            raise TypeError("operation must be callable")
        return [operation(value) for value in values]


@dataclass(frozen=True, slots=True)
class VirtualGPU:
    """Parallel-style matrix execution implemented in software."""

    name: str = "ARIA Virtual GPU"

    def matmul(
        self,
        left: Sequence[Sequence[T]],
        right: Sequence[Sequence[T]],
    ) -> list[list[T]]:
        _, left_width = _matrix_shape(left, "left matrix")
        _, right_width = _matrix_shape(right, "right matrix")

        if len(right) != left_width:
            raise ValueError("matrix dimensions do not align")

        return [
            [
                sum(left[row][k] * right[k][column] for k in range(left_width))
                for column in range(right_width)
            ]
            for row in range(len(left))
        ]


@dataclass(frozen=True, slots=True)
class VirtualNPU:
    """Neural-network-oriented software execution."""

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

        if not result:
            raise ValueError("dense input must not be empty")
        if len(result[0]) != len(bias):
            raise ValueError("bias width does not match dense output")

        return [
            [value + bias[column] for column, value in enumerate(row)]
            for row in result
        ]

    def relu(self, values: Sequence[float]) -> list[float]:
        return [max(0.0, value) for value in values]
