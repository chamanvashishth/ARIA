"""Minimal model/module abstraction."""

from __future__ import annotations

from aria.brain.parameter import Parameter


class Module:
    """Base class for ARIA neural components."""

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

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()
