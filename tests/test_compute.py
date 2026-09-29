import pytest

from aria.compute import ComputeKind, ComputeTask, VirtualCPU, VirtualGPU, VirtualNPU
from aria.runtime import ComputeDispatcher


def test_virtual_cpu_maps_values():
    cpu = VirtualCPU()
    assert cpu.map([1, 2, 3], lambda value: value * 2) == [2, 4, 6]


def test_virtual_gpu_matrix_multiplication():
    gpu = VirtualGPU()
    assert gpu.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [
        [19, 22],
        [43, 50],
    ]


def test_virtual_npu_dense_and_relu():
    npu = VirtualNPU()
    assert npu.dense([[1.0, 2.0]], [[2.0, 1.0], [1.0, 3.0]], [1.0, -1.0]) == [
        [5.0, 6.0]
    ]
    assert npu.relu([-2.0, 0.0, 3.0]) == [0.0, 0.0, 3.0]


def test_task_validation():
    with pytest.raises(ValueError):
        ComputeTask("", ComputeKind.CPU, lambda engine: engine)

    with pytest.raises(TypeError):
        ComputeTask("demo", ComputeKind.CPU, None)


def test_dispatcher_routes_to_the_declared_virtual_engine():
    runtime = ComputeDispatcher()
    task = ComputeTask(
        "double",
        ComputeKind.CPU,
        lambda engine, values: engine.map(values, lambda value: value * 2),
    )
    assert runtime.execute(task, [2, 4, 6]) == [4, 8, 12]


def test_dispatcher_exposes_only_aria_engines():
    runtime = ComputeDispatcher()
    assert runtime.engine(ComputeKind.CPU).name == "ARIA Virtual CPU"
    assert runtime.engine(ComputeKind.GPU).name == "ARIA Virtual GPU"
    assert runtime.engine(ComputeKind.NPU).name == "ARIA Virtual NPU"
