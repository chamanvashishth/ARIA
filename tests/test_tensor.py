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
