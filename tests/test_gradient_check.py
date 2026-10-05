import pytest

from aria.brain import Parameter, Tensor
from aria.evaluation import check_gradients


def test_gradient_check_matches_finite_difference_for_quadratic() -> None:
    parameter = Parameter([1.5, -2.0, 0.25])

    report = check_gradients(
        lambda: (parameter * parameter).sum(),
        [parameter],
        max_checks=None,
    )

    assert report.passed
    assert report.checked_values == 3
    assert report.mismatched_values == 0
    assert report.max_absolute_error < 1e-5
    assert parameter._values == [1.5, -2.0, 0.25]


def test_gradient_check_respects_max_checks() -> None:
    parameter = Parameter([1.0, 2.0, 3.0])

    report = check_gradients(
        lambda: (parameter * parameter).sum(),
        [parameter],
        max_checks=2,
    )

    assert report.passed
    assert report.checked_values == 2


def test_gradient_check_rejects_invalid_options() -> None:
    parameter = Parameter(2.0)

    with pytest.raises(ValueError, match="epsilon"):
        check_gradients(lambda: parameter * parameter, [parameter], epsilon=0.0)

    with pytest.raises(ValueError, match="scalar tensor"):
        check_gradients(lambda: Tensor.from_list([1.0]), [parameter])
