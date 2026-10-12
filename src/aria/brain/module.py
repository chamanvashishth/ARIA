"""Minimal model/module abstraction."""

from __future__ import annotations

from typing import Any

from aria.brain.parameter import Parameter


class Module:
    """Base class for ARIA neural components."""

    def forward(self, inputs: Any) -> Any:
        """Run a forward pass; concrete modules implement their input contract."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def requires_grad_(self, requires_grad: bool) -> Module:
        """Enable or disable gradient tracking for every parameter in this module."""
        for parameter in self.parameters():
            parameter.requires_grad = requires_grad
        return self

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()
