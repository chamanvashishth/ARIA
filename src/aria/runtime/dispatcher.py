"""Dispatch compute tasks to ARIA-owned virtual engines."""

from aria.compute import ComputeKind, ComputeTask, VirtualCPU, VirtualGPU, VirtualNPU


class ComputeDispatcher:
    """Route explicit tasks without depending on physical accelerators."""

    def __init__(
        self,
        cpu: VirtualCPU | None = None,
        gpu: VirtualGPU | None = None,
        npu: VirtualNPU | None = None,
    ) -> None:
        self._engines = {
            ComputeKind.CPU: cpu or VirtualCPU(),
            ComputeKind.GPU: gpu or VirtualGPU(),
            ComputeKind.NPU: npu or VirtualNPU(),
        }

    def execute(self, task: ComputeTask, *args, **kwargs):
        engine = self._engines[task.kind]
        return task.operation(engine, *args, **kwargs)

    def engine(self, kind: ComputeKind):
        return self._engines[kind]
