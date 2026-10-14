import pytest

from aria.brain import Parameter, ReLU, Tensor


def test_tensor_graph_trace_reports_operation_dependencies() -> None:
    left = Parameter([[1.0, 2.0]])
    right = Tensor([[3.0], [4.0]])
    product = left.matmul(right)
    activated = ReLU().forward(product)
    loss = activated.sum()
    trace = loss.graph_trace()

    assert trace.validate()
    assert trace.output_id == trace.nodes[-1].node_id
    assert trace.operation_counts == {
        "leaf": 2, "matmul": 1, "relu": 1, "sum": 1
    }
    matmul_node = next(node for node in trace.nodes if node.operation == "matmul")
    assert len(matmul_node.dependencies) == 2
    assert matmul_node.requires_grad
    assert matmul_node.has_backward


def test_tensor_graph_trace_includes_elementwise_operations_and_reshape() -> None:
    value = Parameter([[1.0, 2.0]])
    result = ((value + 2.0) * value).reshape((2,)).sum()
    trace = result.graph_trace()

    assert trace.validate()
    assert trace.operation_counts == {
        "leaf": 2, "add": 1, "multiply": 1, "reshape": 1, "sum": 1
    }
    assert all(node.shape for node in trace.nodes if node.operation != "sum")
    assert trace.nodes[-1].operation == "sum"


def test_tensor_graph_trace_does_not_change_backward_gradients() -> None:
    value = Parameter([[2.0, -3.0]])
    loss = (value * value).sum()
    assert loss.graph_trace().validate()
    loss.backward()
    assert value.grad is not None
    assert value.grad.to_list() == [[4.0, -6.0]]


def test_tensor_graph_trace_rejects_non_tensor_output() -> None:
    from aria.compute import TensorGraphTrace
    with pytest.raises(TypeError, match="must be a Tensor"):
        TensorGraphTrace.from_tensor([[1.0]])
