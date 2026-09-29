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
