"""Software-emulated compute devices for ARIA."""

from aria.compute.backends import (
    BackendInfo,
    ComputeBackend,
    ComputeRuntime,
    DeviceKind,
    MatmulProfile,
    Matrix,
    VirtualCPUBackend,
    VirtualGPUBackend,
    VirtualNPUBackend,
)

__all__ = [
    "BackendInfo",
    "ComputeBackend",
    "ComputeRuntime",
    "DeviceKind",
    "MatmulProfile",
    "Matrix",
    "VirtualCPUBackend",
    "VirtualGPUBackend",
    "VirtualNPUBackend",
]
