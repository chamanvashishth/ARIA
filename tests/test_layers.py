import math

import pytest

from aria.brain import Embedding, Linear, Parameter, ReLU, Sequential, Tensor


def test_linear_forward_shape_and_bias() -> None:
    layer = Linear(2, 2, seed=7)
    layer.weight = Parameter([[1.0, 2.0], [3.0, 4.0]])
    layer.bias = Parameter([1.0, -1.0])

    output = layer.forward(Tensor.from_list([[2.0, 3.0]]))

    assert output.shape == (1, 2)
    assert output.to_list() == [[12.0, 15.0]]


def test_relu_forward() -> None:
    output = ReLU().forward(Tensor.from_list([[-2.0, 0.0, 3.0]]))
    assert output.to_list() == [[0.0, 0.0, 3.0]]


def test_embedding_lookup() -> None:
    layer = Embedding(4, 2)
    layer.weight = Parameter([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])

    output = layer.forward([2, 0])
    assert output.to_list() == [[4.0, 5.0], [0.0, 1.0]]


def test_linear_backpropagates_to_parameters() -> None:
    layer = Linear(2, 1)
    layer.weight = Parameter([[2.0], [3.0]])
    layer.bias = Parameter([1.0])

    loss = layer.forward(Tensor.from_list([[4.0, 5.0]])).sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.weight.grad.to_list() == [[4.0], [5.0]]
    assert layer.bias.grad.to_list() == [1.0]


def test_embedding_backpropagates_to_used_rows() -> None:
    layer = Embedding(3, 2)
    layer.weight = Parameter([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])

    loss = layer.forward([2, 0, 2]).sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.to_list() == [[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]]


def test_embedding_rejects_invalid_token() -> None:
    with pytest.raises(IndexError):
        Embedding(2, 3).forward([2])


def test_sequential_composes_layers() -> None:
    model = Sequential(Linear(2, 3, seed=1), ReLU(), Linear(3, 1, seed=2))
    output = model.forward(Tensor.from_list([[1.0, 2.0]]))
    assert output.shape == (1, 1)



def test_relu_input_gradients_match_finite_differences_away_from_zero() -> None:
    from aria.evaluation.core import check_gradients

    inputs = Parameter([[-1.5, 0.25, 2.0]])
    report = check_gradients(
        lambda: (ReLU().forward(inputs) * ReLU().forward(inputs)).sum(),
        [inputs],
        epsilon=1e-5,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
        max_checks=None,
    )

    assert report.passed
    assert report.checked_values == len(inputs._values)
    assert report.max_absolute_error < 1e-5


def test_embedding_parameter_gradients_match_finite_differences_with_repeated_ids() -> None:
    from aria.evaluation.core import check_gradients

    layer = Embedding(3, 2, seed=85)
    token_ids = [2, 0, 2]
    report = check_gradients(
        lambda: (layer.forward(token_ids) * layer.forward(token_ids)).sum(),
        layer.parameters(),
        epsilon=1e-5,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
        max_checks=None,
    )

    assert report.passed
    assert report.checked_values == len(layer.weight._values)
    assert report.max_absolute_error < 1e-5


def test_embedding_accumulates_repeated_token_rows_in_backward() -> None:
    layer = Embedding(3, 2, seed=86)
    layer.weight = Parameter([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    layer.forward([2, 2, 0]).sum().backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.to_list() == [
        [1.0, 1.0],
        [0.0, 0.0],
        [2.0, 2.0],
    ]



@pytest.mark.parametrize("backend_name", ["vcpu", "vgpu"])
def test_linear_backend_forward_and_gradients(backend_name: str) -> None:
    from aria.compute import ComputeRuntime

    layer = Linear(2, 2, seed=202, backend=ComputeRuntime(backend_name))
    inputs = Parameter([[0.5, -1.0], [2.0, 0.25]])
    loss = layer.forward(inputs).sum()
    loss.backward()

    assert inputs.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias is not None and layer.bias.grad is not None
    assert all(math.isfinite(value) for value in inputs.grad._values)
    assert all(math.isfinite(value) for value in layer.weight.grad._values)


def test_linear_preserves_leading_dimensions_with_backend() -> None:
    from aria.compute import VirtualGPUBackend

    layer = Linear(2, 3, seed=203, backend=VirtualGPUBackend(tile_size=2))
    output = layer.forward(Tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    assert output.shape == (1, 2, 3)


def test_linear_vnpu_is_forward_only_and_requires_frozen_parameters() -> None:
    from aria.compute import VirtualNPUBackend

    layer = Linear(2, 2, seed=204, backend=VirtualNPUBackend())
    inputs = Tensor([[0.25, -0.5]])
    with pytest.raises(RuntimeError, match="inference-only"):
        layer.forward(inputs)

    layer.requires_grad_(False)
    output = layer.forward(inputs)
    assert output.shape == (1, 2)
    assert all(math.isfinite(value) for value in output._values)
    assert output.requires_grad is False



def test_linear_bias_operation_is_visible_in_tensor_trace() -> None:
    layer = Linear(2, 3, seed=301)
    output = layer.forward(Tensor([[1.0, 2.0]]))
    assert output.graph_trace().operation_counts["bias_add"] == 1


def test_sequential_graph_execution_matches_eager_output() -> None:
    model = Sequential(Linear(2, 3, seed=401), ReLU(), Linear(3, 1, seed=402))
    inputs = Tensor.from_list([[1.0, -2.0], [0.5, 3.0]])

    eager_values = model.forward(inputs)._values.copy()
    execution = model.graph_execute(inputs)

    assert [row[0] for row in execution.output_values] == pytest.approx(eager_values)
    assert all(metric.state == "completed" for metric in execution.graph_result.node_metrics)
    assert execution.trace.operation_counts["matmul"] == 2
    assert execution.trace.operation_counts["relu"] == 1
    assert execution.trace.operation_counts["bias_add"] == 2


def test_sequential_graph_training_backpropagates_and_updates_all_layers() -> None:
    from aria.brain import SGD

    first = Linear(2, 2, seed=403)
    first.weight = Parameter([[0.5, -0.25], [0.75, 0.5]])
    first.bias = Parameter([0.1, 0.2])
    second = Linear(2, 1, seed=404)
    second.weight = Parameter([[0.4], [-0.3]])
    second.bias = Parameter([0.05])
    model = Sequential(first, ReLU(), second)
    inputs = Parameter([[1.0, 2.0], [2.0, -1.0]])
    initial_parameters = [parameter._values.copy() for parameter in model.parameters()]

    execution = model.graph_execute(
        inputs,
        objective=lambda output: output.sum(),
        max_workers=2,
    )
    assert isinstance(execution.output_values, float)

    execution.backward()
    parameters = model.parameters()
    assert all(parameter.grad is not None for parameter in parameters)
    assert inputs.grad is not None

    optimizer = SGD(parameters, learning_rate=0.05)
    optimizer.step()
    assert all(
        parameter._values != before
        for parameter, before in zip(parameters, initial_parameters)
    )
