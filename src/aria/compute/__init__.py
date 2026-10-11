"""Software-emulated compute devices for ARIA."""

from aria.compute.backends import (
    BackendInfo,
    ComputeBackend,
    ComputeRuntime,
    DeviceKind,
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
    "Matrix",
    "VirtualCPUBackend",
    "VirtualGPUBackend",
    "VirtualNPUBackend",
]
