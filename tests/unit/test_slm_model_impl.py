"""Unit tests for the baseline SLM model implementation."""

import pytest

from aria.slm import EmbeddingProjectionModel, SLMConfig


@pytest.fixture
def config() -> SLMConfig:
    return SLMConfig(
        vocab_size=4,
        context_length=3,
        embedding_dim=4,
        num_layers=1,
        num_heads=1,
    )


def test_baseline_model_produces_expected_output_shape(config: SLMConfig) -> None:
    model = EmbeddingProjectionModel(config, seed=7)

    output = model.forward(((0, 1, 2),))

    assert output.batch_size == 1
    assert output.sequence_length == 3
    assert output.vocab_size == 4


def test_same_seed_produces_deterministic_logits(config: SLMConfig) -> None:
    first = EmbeddingProjectionModel(config, seed=7).forward(((0, 1),))
    second = EmbeddingProjectionModel(config, seed=7).forward(((0, 1),))

    assert first == second


def test_model_rejects_invalid_token_ids(config: SLMConfig) -> None:
    model = EmbeddingProjectionModel(config)

    with pytest.raises(ValueError, match="outside vocab_size"):
        model.forward(((4,),))

    with pytest.raises(TypeError, match="must be integers"):
        model.forward(((True,),))


def test_model_rejects_sequences_longer_than_context(config: SLMConfig) -> None:
    model = EmbeddingProjectionModel(config)

    with pytest.raises(ValueError, match="context_length"):
        model.forward(((0, 1, 2, 3),))
