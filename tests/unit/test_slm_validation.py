"""Unit tests for SLM training-data validation."""

import pytest

from aria.slm import SLMConfig, TrainingExample, validate_examples


@pytest.fixture
def config() -> SLMConfig:
    return SLMConfig(
        vocab_size=8,
        context_length=2,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
    )


def test_validate_examples_accepts_valid_examples(config: SLMConfig) -> None:
    examples = (
        TrainingExample((0, 1), (1, 2)),
        TrainingExample((2, 3), (3, 4)),
    )

    assert validate_examples(examples, config) == 2


def test_validate_examples_rejects_wrong_context_length(config: SLMConfig) -> None:
    examples = (TrainingExample((0,), (1,)),)

    with pytest.raises(ValueError, match="context_length"):
        validate_examples(examples, config)


def test_validate_examples_rejects_out_of_range_ids(config: SLMConfig) -> None:
    examples = (TrainingExample((0, 7), (7, 8)),)

    with pytest.raises(ValueError, match="outside vocab_size"):
        validate_examples(examples, config)


def test_validate_examples_rejects_wrong_example_type(config: SLMConfig) -> None:
    with pytest.raises(TypeError, match="must be a TrainingExample"):
        validate_examples((object(),), config)
