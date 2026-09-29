"""Unit tests for the SLM model architecture contract."""

from aria.slm import ModelOutput, SLMModel


class DummyModel:
    def forward(self, input_ids: tuple[tuple[int, ...], ...]) -> ModelOutput:
        return ModelOutput(
            logits=(((0.1, 0.2), (0.3, 0.4)),),
        )


def test_model_output_exposes_expected_dimensions() -> None:
    output = ModelOutput(logits=(((0.1, 0.2), (0.3, 0.4)),))

    assert output.batch_size == 1
    assert output.sequence_length == 2
    assert output.vocab_size == 2


def test_model_contract_accepts_forward_implementation() -> None:
    model = DummyModel()

    assert isinstance(model, SLMModel)
    output = model.forward(((1, 2),))
    assert output.sequence_length == 2
    assert output.vocab_size == 2
