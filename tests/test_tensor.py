import pytest

from aria.brain import Module, Parameter, Tensor


def test_tensor_add_and_multiply_forward() -> None:
    x = Tensor.from_list([1.0, 2.0], requires_grad=True)
    y = Tensor.from_list([3.0, 4.0], requires_grad=True)
    result = (x + y) * y
    assert result.to_list() == [12.0, 24.0]


def test_scalar_backward_accumulates_gradients() -> None:
    x = Parameter(2.0)
    y = Parameter(3.0)
    loss = (x * y).sum()
    loss.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert x.grad.item() == pytest.approx(3.0)
    assert y.grad.item() == pytest.approx(2.0)


def test_module_collects_parameters() -> None:
    class Linear(Module):
        def __init__(self) -> None:
            self.weight = Parameter([1.0, 2.0])
            self.bias = Parameter([0.0, 0.0])

    assert len(Linear().parameters()) == 2


def test_rejects_ragged_tensor() -> None:
    with pytest.raises(ValueError):
        Tensor.from_list([[1.0], [2.0, 3.0]])



def test_repeated_backward_reuses_graph_without_stale_intermediate_gradients() -> None:
    x = Parameter(2.0)
    squared = x * x
    loss = squared.sum()

    loss.backward()
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(4.0)

    loss.backward()
    assert x.grad is not None
    # Leaf gradients accumulate, but the intermediate squared gradient must
    # be recomputed instead of retaining its value from the previous traversal.
    assert x.grad.item() == pytest.approx(8.0)



def test_shared_intermediate_accumulates_gradients_from_both_branches() -> None:
    x = Parameter(2.0)
    shared = x * x
    left = shared * 3.0
    right = shared * 5.0
    loss = (left + right).sum()

    loss.backward()

    assert shared.grad is not None
    assert shared.grad.item() == pytest.approx(8.0)
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(32.0)


def test_repeated_backward_on_branching_graph_accumulates_only_leaf_gradients() -> None:
    x = Parameter(2.0)
    shared = x * x
    loss = (shared * 3.0 + shared * 5.0).sum()

    loss.backward()
    loss.backward()

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(64.0)



def test_linear_layer_parameter_gradients_match_finite_differences() -> None:
    from aria.brain.layers import Linear
    from aria.evaluation.core import check_gradients

    layer = Linear(2, 2, seed=83)
    inputs = Tensor([[0.25, -0.4], [0.7, 0.15]])

    report = check_gradients(
        lambda: (layer.forward(inputs) * layer.forward(inputs)).sum(),
        layer.parameters(),
        epsilon=1e-5,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
        max_checks=None,
    )

    assert report.passed
    assert report.checked_values == sum(len(parameter._values) for parameter in layer.parameters())
    assert report.max_absolute_error < 1e-5


def test_linear_without_bias_propagates_input_gradients() -> None:
    from aria.brain.layers import Linear

    layer = Linear(2, 1, bias=False, seed=84)
    inputs = Tensor([[0.5, -0.25]], requires_grad=True)
    layer.forward(inputs).sum().backward()

    assert inputs.grad is not None
    assert len(inputs.grad._values) == len(inputs._values)
    assert layer.bias is None
    assert layer.weight.grad is not None


def test_gradient_values_requires_backward_and_returns_accumulated_gradient() -> None:
    from aria.brain.tensor import Tensor

    value = Tensor.scalar(3.0, requires_grad=True)
    with pytest.raises(RuntimeError, match="gradient is not available"):
        value.gradient_values()

    result = (value * value).sum()
    result.backward()

    assert result.gradient_values() == [1.0]
    assert value.gradient_values() == [6.0]



def test_tensor_matmul_forward_and_reverse_mode_gradients() -> None:
    from aria.compute import VirtualCPUBackend

    left = Parameter([[1.0, 2.0], [3.0, 4.0]])
    right = Parameter([[5.0, 6.0], [7.0, 8.0]])
    output = left.matmul(right, backend=VirtualCPUBackend())

    assert output.to_list() == [[19.0, 22.0], [43.0, 50.0]]
    output.sum().backward()
    assert left.grad is not None
    assert left.grad.to_list() == [[11.0, 15.0], [11.0, 15.0]]
    assert right.grad is not None
    assert right.grad.to_list() == [[4.0, 4.0], [6.0, 6.0]]


def test_tensor_matmul_vgpu_gradients_match_cpu_reference() -> None:
    from aria.compute import VirtualCPUBackend, VirtualGPUBackend

    left_data = [[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]]
    right_data = [[1.0, 2.0], [-2.0, 0.5], [0.25, -1.0]]
    cpu_left, cpu_right = Parameter(left_data), Parameter(right_data)
    gpu_left, gpu_right = Parameter(left_data), Parameter(right_data)
    cpu_loss = cpu_left.matmul(cpu_right, backend=VirtualCPUBackend()).sum()
    gpu_loss = gpu_left.matmul(gpu_right, backend=VirtualGPUBackend(tile_size=1)).sum()
    cpu_loss.backward()
    gpu_loss.backward()

    assert gpu_loss.item() == pytest.approx(cpu_loss.item(), abs=1e-12)
    assert gpu_left.gradient_values() == pytest.approx(cpu_left.gradient_values())
    assert gpu_right.gradient_values() == pytest.approx(cpu_right.gradient_values())


def test_tensor_matmul_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="both be 2-D"):
        Tensor([1.0, 2.0]).matmul(Tensor([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="dimensions do not match"):
        Tensor([[1.0, 2.0]]).matmul(Tensor([[1.0, 2.0]]))


def test_tensor_matmul_vnpu_rejects_gradient_tracking() -> None:
    from aria.compute import VirtualNPUBackend

    with pytest.raises(RuntimeError, match="inference-only"):
        Parameter([[1.0]]).matmul(Tensor([[2.0]]), backend=VirtualNPUBackend())
