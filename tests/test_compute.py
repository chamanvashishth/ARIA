import math

import pytest

from aria.compute import (
    ComputeRuntime,
    MatmulProfile,
    VirtualCPUBackend,
    VirtualGPUBackend,
    VirtualNPUBackend,
)


def test_virtual_cpu_matrix_multiplication() -> None:
    backend = VirtualCPUBackend()
    assert backend.matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]) == [
        [58.0, 64.0],
        [139.0, 154.0],
    ]


@pytest.mark.parametrize("tile_size", [1, 2, 4, 16])
def test_virtual_gpu_tiled_result_matches_cpu_reference(tile_size: int) -> None:
    left = [[1.0, -2.0, 3.5], [4.0, 5.0, -6.0], [0.25, 8.0, 2.0]]
    right = [[7.0, 1.0], [9.0, -3.0], [11.0, 2.0]]

    expected = VirtualCPUBackend().matmul(left, right)
    actual = VirtualGPUBackend(tile_size=tile_size).matmul(left, right)

    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row, rel=1e-12, abs=1e-12)


def test_virtual_npu_quantized_result_is_close_to_float_reference() -> None:
    left = [[0.25, -0.5, 1.0], [1.5, 0.75, -0.25]]
    right = [[1.0, 0.5], [-0.5, 1.25], [0.25, -1.0]]

    expected = VirtualCPUBackend().matmul(left, right)
    actual = VirtualNPUBackend().matmul(left, right)

    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row, rel=0.02, abs=0.02)


def test_virtual_npu_handles_zero_matrices() -> None:
    assert VirtualNPUBackend().matmul([[0, 0]], [[0], [0]]) == [[0.0]]


def test_compute_runtime_selects_explicit_device() -> None:
    runtime = ComputeRuntime("vgpu", tile_size=2)

    assert runtime.info.device == "vgpu"
    assert runtime.info.hardware_acceleration is False
    assert runtime.matmul([[2, 3]], [[4], [5]]) == [[23.0]]
    assert {item.device for item in runtime.available_devices()} == {
        "vcpu", "vgpu", "vnpu"
    }


def test_virtual_devices_are_explicitly_marked_as_emulators() -> None:
    cpu, gpu, npu = ComputeRuntime.available_devices()

    assert cpu.execution == "pure-python-reference"
    assert gpu.execution.startswith("host-cpu-tiled-emulation")
    assert npu.execution == "host-cpu-int8-emulation"
    assert all(not item.hardware_acceleration for item in (cpu, gpu, npu))
    assert cpu.supports_autodiff is True
    assert gpu.supports_autodiff is True
    assert npu.supports_autodiff is False


@pytest.mark.parametrize("device", ["unknown", "cuda", "metal"])
def test_compute_runtime_rejects_unknown_device(device: str) -> None:
    with pytest.raises(ValueError, match="unsupported compute device"):
        ComputeRuntime(device)  # type: ignore[arg-type]


@pytest.mark.parametrize("tile_size", [0, -1, True, 1.5])
def test_virtual_gpu_rejects_invalid_tile_size(tile_size: object) -> None:
    with pytest.raises(ValueError, match="tile_size must be a positive integer"):
        VirtualGPUBackend(tile_size=tile_size)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("left", "right", "error", "message"),
    [
        ([], [[1]], ValueError, "left operand must not be empty"),
        ([[1]], [], ValueError, "right operand must not be empty"),
        ([[1, 2]], [[1]], ValueError, "matrix dimensions do not match"),
        ([[1], [2, 3]], [[1]], ValueError, "left operand must be rectangular"),
        ([[math.inf]], [[1]], ValueError, "values must be finite"),
        ([[True]], [[1]], TypeError, "values must be real numbers"),
    ],
)
def test_compute_backends_validate_matrix_inputs(left, right, error, message) -> None:
    for backend in (VirtualCPUBackend(), VirtualGPUBackend(), VirtualNPUBackend()):
        with pytest.raises(error, match=message):
            backend.matmul(left, right)


def test_runtime_profiles_successful_operations_when_enabled() -> None:
    runtime = ComputeRuntime("vcpu", profiling=True)

    assert runtime.matmul([[1, 2]], [[3], [4]]) == [[11.0]]
    assert runtime.profiles == (
        MatmulProfile(
            device="vcpu",
            left_shape=(1, 2),
            right_shape=(2, 1),
            elapsed_ns=runtime.profiles[0].elapsed_ns,
            succeeded=True,
        ),
    )
    assert runtime.profiles[0].elapsed_ns >= 0
    assert runtime.profiles[0].error_type is None


def test_runtime_profiles_failed_operations_and_reraises_error() -> None:
    runtime = ComputeRuntime("vgpu", profiling=True)

    with pytest.raises(ValueError, match="matrix dimensions do not match"):
        runtime.matmul([[1, 2]], [[3]])

    assert len(runtime.profiles) == 1
    profile = runtime.profiles[0]
    assert profile.device == "vgpu"
    assert profile.left_shape == (1, 2)
    assert profile.right_shape == (1, 1)
    assert profile.succeeded is False
    assert profile.error_type == "ValueError"
    assert profile.elapsed_ns >= 0


def test_runtime_profiling_is_opt_in_and_profiles_can_be_cleared() -> None:
    runtime = ComputeRuntime("vcpu")
    runtime.matmul([[2]], [[5]])
    assert runtime.profiles == ()

    profiled = ComputeRuntime("vcpu", profiling=True)
    profiled.matmul([[2]], [[5]])
    assert len(profiled.profiles) == 1
    profiled.clear_profiles()
    assert profiled.profiles == ()
