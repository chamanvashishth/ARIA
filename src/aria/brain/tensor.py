"""Small dependency-free tensor primitive with reverse-mode autodiff."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable, Sequence


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
    return [float(data)]


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
    ) -> Tensor:
        out = cls(values, requires_grad=any(p.requires_grad for p in parents), _parents=parents)
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
        )

        def backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad._values)
            if rhs.requires_grad:
                rhs._accumulate(out.grad._values)

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
        )

        def backward() -> None:
            if self.requires_grad:
                self._accumulate([g * b for g, b in zip(out.grad._values, rhs._values)])
            if rhs.requires_grad:
                rhs._accumulate([g * a for g, a in zip(out.grad._values, self._values)])

        out._backward = backward
        return out

    def sum(self) -> Tensor:
        def backward(out: Tensor) -> None:
            if self.requires_grad:
                scale = out.grad._values[0]
                self._accumulate([scale] * len(self._values))

        return Tensor.operation(sum(self._values), parents=(self,), backward=backward)


    def backward(self) -> None:
        if self.shape != ():
            raise ValueError("backward() requires a scalar tensor")
        if not self.requires_grad:
            raise ValueError("cannot backpropagate from a tensor without gradients")

        topo: list[Tensor] = []
        seen: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                visit(parent)
            topo.append(node)

        visit(self)
        self.grad = Tensor.scalar(1.0)

        for node in reversed(topo):
            if node._backward is not None:
                node._backward()

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
            backward=lambda result: self._accumulate(result.grad._values) if self.requires_grad else None,
        )
        return out

    def to_list(self) -> object:
        return _build(self._values, self.shape)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
