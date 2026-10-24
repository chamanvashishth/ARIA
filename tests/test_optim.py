import math

import pytest

from aria.brain import Parameter, SGD
from aria.brain.tensor import Tensor


def _set_gradient(parameter: Parameter, values: object) -> None:
    parameter.grad = Tensor(values)


def test_sgd_rejects_non_finite_gradient_without_mutating_any_parameter() -> None:
    first = Parameter([1.0, 2.0])
    second = Parameter([3.0])
    _set_gradient(first, [0.5, 0.5])
    _set_gradient(second, [math.nan])
    optimizer = SGD([first, second], learning_rate=0.1)

    with pytest.raises(ValueError, match="gradient values must be finite"):
        optimizer.step()

    assert first._values == [1.0, 2.0]
    assert first.data == [1.0, 2.0]
    assert second._values == [3.0]


def test_sgd_rejects_update_that_would_overflow_without_mutating_parameters() -> None:
    parameter = Parameter([1.0])
    _set_gradient(parameter, [2.0])
    optimizer = SGD([parameter], learning_rate=0.1)
    optimizer.learning_rate = 1e308

    with pytest.raises(ValueError, match="non-finite parameter values"):
        optimizer.step()

    assert parameter._values == [1.0]
    assert parameter.data == [1.0]


@pytest.mark.parametrize("learning_rate", [0.0, -0.1, math.inf, math.nan])
def test_sgd_revalidates_learning_rate_before_update(learning_rate: float) -> None:
    parameter = Parameter([1.0])
    _set_gradient(parameter, [0.5])
    optimizer = SGD([parameter], learning_rate=0.1)
    optimizer.learning_rate = learning_rate

    with pytest.raises(ValueError, match="learning rate must be finite and positive"):
        optimizer.step()

    assert parameter._values == [1.0]


def test_sgd_rejects_gradient_shape_mismatch_without_mutating_parameters() -> None:
    parameter = Parameter([1.0, 2.0])
    _set_gradient(parameter, [0.5])
    optimizer = SGD([parameter], learning_rate=0.1)

    with pytest.raises(ValueError, match="gradient size must match parameter size"):
        optimizer.step()

    assert parameter._values == [1.0, 2.0]


def test_sgd_skips_parameters_without_gradients() -> None:
    parameter = Parameter([1.0, 2.0])
    optimizer = SGD([parameter], learning_rate=0.1)

    optimizer.step()

    assert parameter._values == [1.0, 2.0]
