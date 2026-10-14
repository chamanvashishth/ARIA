"""Small dependency-free tensor primitive with reverse-mode autodiff."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable, Sequence

from aria.compute.backends import ComputeBackend, VirtualCPUBackend


Shape = tuple[int, ...]


def _shape(data: object) -> Shape:
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    child = _shape(data[0])
    if any(_shape(item) != child for item in data):
        raise ValueError("tensor data must be rectangular")
    return (len(data),) + child


def _flatten(data: object) -> list[float]:
    if isinstance(data, list):
        result: list[float] = []
        for item in data:
            result.extend(_flatten(item))
        return result
    if isinstance(data, (int, float)):
        return [float(data)]
    raise TypeError("tensor values must be numeric")


def _build(values: Sequence[float], shape: Shape) -> object:
    if not shape:
        return float(values[0])
    width = prod(shape[1:])
    return [
        _build(values[i * width:(i + 1) * width], shape[1:])
        for i in range(shape[0])
    ]


@dataclass
class Tensor:
    """Dense tensor with a minimal computation graph.

    This is a correctness-first foundation, not the final high-performance
    tensor backend.
    """

    data: object
    requires_grad: bool = False
    grad: Tensor | None = None
    _parents: tuple[Tensor, ...] = ()
    _backward: Callable[[], None] | None = None
    _operation: str = "leaf"

    def __post_init__(self) -> None:
        self.shape = _shape(self.data)
        self._values = _flatten(self.data)

    @classmethod
    def scalar(cls, value: float, requires_grad: bool = False) -> Tensor:
        return cls(float(value), requires_grad=requires_grad)

    @classmethod
    def operation(
        cls,
        values: object,
        *,
        parents: tuple[Tensor, ...],
        backward: Callable[[Tensor], None],
        operation: str = "custom",
    ) -> Tensor:
        out = cls(
            values,
            requires_grad=any(p.requires_grad for p in parents),
            _parents=parents,
            _operation=operation,
        )
        out._backward = lambda: backward(out)
        return out

    @classmethod
    def zeros(cls, shape: Shape, requires_grad: bool = False) -> Tensor:
        if any(dim < 0 for dim in shape):
            raise ValueError("tensor dimensions must be non-negative")
        return cls(_build([0.0] * prod(shape), shape), requires_grad=requires_grad)

    @classmethod
    def from_list(cls, data: list[object], requires_grad: bool = False) -> Tensor:
        return cls(data, requires_grad=requires_grad)

    def gradient_values(self) -> list[float]:
        """Return this tensor's gradient values, failing clearly before backward()."""
        if self.grad is None:
            raise RuntimeError("gradient is not available before backward()")
        return self.grad._values

    def _accumulate(self, values: Sequence[float]) -> None:
        incoming = Tensor(_build(values, self.shape))
        self.grad = incoming if self.grad is None else self.grad + incoming

    def __add__(self, other: Tensor | float) -> Tensor:
        rhs = other if isinstance(other, Tensor) else Tensor.scalar(other)
        if self.shape != rhs.shape:
            raise ValueError(f"shape mismatch: {self.shape} != {rhs.shape}")
        out = Tensor(
            _build([a + b for a, b in zip(self._values, rhs._values)], self.shape),
            requires_grad=self.requires_grad or rhs.requires_grad,
            _parents=(self, rhs),
            _operation="add",
        )

        def backward() -> None:
            if self.requires_grad:
                self._accumulate(out.gradient_values())
            if rhs.requires_grad:
                rhs._accumulate(out.gradient_values())

        out._backward = backward
        return out

    def __mul__(self, other: Tensor | float) -> Tensor:
        rhs = other if isinstance(other, Tensor) else Tensor.scalar(other)
        if self.shape != rhs.shape:
            raise ValueError(f"shape mismatch: {self.shape} != {rhs.shape}")
        out = Tensor(
            _build([a * b for a, b in zip(self._values, rhs._values)], self.shape),
            requires_grad=self.requires_grad or rhs.requires_grad,
            _parents=(self, rhs),
            _operation="multiply",
        )

        def backward() -> None:
            if self.requires_grad:
                self._accumulate([g * b for g, b in zip(out.gradient_values(), rhs._values)])
            if rhs.requires_grad:
                rhs._accumulate([g * a for g, a in zip(out.gradient_values(), self._values)])

        out._backward = backward
        return out


    def matmul(self, other: Tensor, *, backend: ComputeBackend | None = None) -> Tensor:
        """Multiply two rank-2 tensors with backend-aware reverse-mode gradients.

        VCPU and VGPU support autodiff. VNPU is quantized inference-only and
        rejects operands that require gradients.
        """
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul operands must both be 2-D tensors")
        rows, inner = self.shape
        other_inner, columns = other.shape
        if inner != other_inner:
            raise ValueError(
                f"matmul dimensions do not match: {self.shape} cannot multiply {other.shape}"
            )

        selected_backend = backend if backend is not None else VirtualCPUBackend()
        if not selected_backend.info.supports_autodiff and (
            self.requires_grad or other.requires_grad
        ):
            raise RuntimeError(
                f"{selected_backend.info.device} backend is inference-only and "
                "cannot be used with tensors that require gradients; freeze model "
                "parameters with Module.requires_grad_(False) for inference"
            )

        left_values = [
            self._values[row * inner:(row + 1) * inner]
            for row in range(rows)
        ]
        right_values = [
            other._values[row * columns:(row + 1) * columns]
            for row in range(other_inner)
        ]
        output_values = selected_backend.matmul(left_values, right_values)

        def backward(out: Tensor) -> None:
            gradient = [
                out.gradient_values()[row * columns:(row + 1) * columns]
                for row in range(rows)
            ]
            if self.requires_grad:
                right_transposed = [
                    [right_values[row][column] for row in range(other_inner)]
                    for column in range(columns)
                ]
                left_gradient = selected_backend.matmul(gradient, right_transposed)
                self._accumulate([value for row in left_gradient for value in row])
            if other.requires_grad:
                left_transposed = [
                    [left_values[row][column] for row in range(rows)]
                    for column in range(inner)
                ]
                right_gradient = selected_backend.matmul(left_transposed, gradient)
                other._accumulate([value for row in right_gradient for value in row])

        return Tensor.operation(
            output_values, parents=(self, other), backward=backward, operation="matmul"
        )

    def sum(self) -> Tensor:
        def backward(out: Tensor) -> None:
            if self.requires_grad:
                scale = out.gradient_values()[0]
                self._accumulate([scale] * len(self._values))

        return Tensor.operation(
            sum(self._values), parents=(self,), backward=backward, operation="sum"
        )


    def backward(self) -> None:
        """Run reverse-mode autodiff through this tensor's recorded graph."""
        self.graph_trace().backward()

    def zero_grad(self) -> None:
        self.grad = None

    def item(self) -> float:
        if self.shape != ():
            raise ValueError("item() requires a scalar tensor")
        return self._values[0]

    def reshape(self, shape: Shape) -> Tensor:
        """Return a view-like tensor with the same values and a gradient path."""
        if any(dim < 0 for dim in shape):
            raise ValueError("tensor dimensions must be non-negative")
        if prod(shape) != len(self._values):
            raise ValueError("reshape must preserve the number of values")
        out = Tensor.operation(
            _build(self._values, shape),
            parents=(self,),
            backward=lambda result: self._accumulate(result.gradient_values()) if self.requires_grad else None,
            operation="reshape",
        )
        return out

    def graph_trace(self):
        """Return an immutable, inspectable trace of this tensor's autograd graph.

        Tracing does not rerun the forward pass or replace reverse-mode autodiff;
        it exposes the operation/dependency structure already recorded by Tensor.
        """
        from aria.compute.graph import TensorGraphTrace

        return TensorGraphTrace.from_tensor(self)

    def to_list(self) -> object:
        return _build(self._values, self.shape)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
