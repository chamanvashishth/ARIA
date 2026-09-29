from aria.compute import ComputeKind, ComputeTask, VirtualCPU, VirtualGPU, VirtualNPU


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


def test_compute_task_declares_owned_execution_kind():
    task = ComputeTask("demo", lambda: 1, ComputeKind.NPU)
    assert task.kind is ComputeKind.NPU
