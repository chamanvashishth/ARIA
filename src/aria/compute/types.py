"""Types shared by ARIA's virtual compute layer."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ComputeKind(StrEnum):
    CPU = "vcpu"
    GPU = "vgpu"
    NPU = "vnpu"


@dataclass(frozen=True, slots=True)
class ComputeTask:
    name: str
    kind: ComputeKind
    operation: Callable[..., Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("task name must not be empty")
        if not callable(self.operation):
            raise TypeError("task operation must be callable")
