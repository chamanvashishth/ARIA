"""Types shared by ARIA's virtual compute layer."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable


class ComputeKind(StrEnum):
    CPU = "vcpu"
    GPU = "vgpu"
    NPU = "vnpu"


@dataclass(frozen=True)
class ComputeTask:
    name: str
    operation: Callable[..., object]
    kind: ComputeKind
