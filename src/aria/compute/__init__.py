"""Software-emulated compute devices and scheduling primitives for ARIA."""

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
from aria.compute.scheduler import (
    CommandQueue,
    CommandState,
    ComputeCommand,
    ComputeScheduler,
    ScheduledComputeBackend,
    ScheduledExecutionStats,
    SchedulerStats,
    Workgroup,
)

__all__ = [
    "BackendInfo",
    "CommandQueue",
    "CommandState",
    "ComputeBackend",
    "ComputeCommand",
    "ComputeRuntime",
    "ComputeScheduler",
    "DeviceKind",
    "MatmulProfile",
    "Matrix",
    "ScheduledComputeBackend",
    "ScheduledExecutionStats",
    "SchedulerStats",
    "VirtualCPUBackend",
    "VirtualGPUBackend",
    "VirtualNPUBackend",
    "Workgroup",
]
