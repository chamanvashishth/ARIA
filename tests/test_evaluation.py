import math

import pytest

from aria.brain import TinyLanguageModel
from aria.brain.tensor import Tensor
from aria.evaluation import evaluate_language_model, inspect_parameter_health


def test_evaluation_reports_loss_perplexity_accuracy_and_counts() -> None:
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, seed=4)
    result = evaluate_language_model(model, [([0, 1], [1, 2]), ([2], [3])])

    assert result.mean_loss > 0
    assert result.perplexity == pytest.approx(math.exp(result.mean_loss))
    assert 0.0 <= result.token_accuracy <= 1.0
    assert result.token_count == 3
    assert result.batch_count == 2


def test_evaluation_does_not_backpropagate_or_update_parameters() -> None:
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, seed=8)
    before = [list(parameter._values) for parameter in model.parameters()]

    evaluate_language_model(model, [([0, 1], [1, 2])])

    assert [parameter.grad for parameter in model.parameters()] == [None] * len(before)
    assert [parameter._values for parameter in model.parameters()] == before


def test_evaluation_rejects_empty_examples() -> None:
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, seed=2)
    with pytest.raises(ValueError, match="at least one example"):
        evaluate_language_model(model, [])


def test_evaluation_rejects_non_finite_logits() -> None:
    class InvalidModel:
        def forward(self, token_ids):
            return Tensor.from_list([[float("nan"), 0.0] for _ in token_ids])

    with pytest.raises(ValueError, match="non-finite logits"):
        evaluate_language_model(InvalidModel(), [([0], [1])])


def test_parameter_health_reports_non_finite_values_and_gradients() -> None:
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, seed=3)
    parameters = model.parameters()
    parameters[0]._values[0] = float("inf")
    parameters[1].grad = Tensor.from_list([float("nan")] + [0.0] * (len(parameters[1]._values) - 1))

    report = inspect_parameter_health(model)

    assert report.parameter_count > 0
    assert report.non_finite_values == 1
    assert report.non_finite_gradients == 1
    assert not report.healthy


def test_parameter_health_accepts_finite_model() -> None:
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, seed=3)
    report = inspect_parameter_health(model)
    assert report.healthy
    assert report.non_finite_values == 0
    assert report.non_finite_gradients == 0
