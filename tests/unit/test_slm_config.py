"""Unit tests for the SLM configuration contract."""

import pytest

from aria.slm import SLMConfig


def test_valid_config_is_immutable() -> None:
    config = SLMConfig(
        vocab_size=32_000,
        context_length=2_048,
        embedding_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
    )

    assert config.embedding_dim // config.num_heads == 64
    with pytest.raises(AttributeError):
        config.embedding_dim = 1024


@pytest.mark.parametrize(
    "field",
    [
        "vocab_size",
        "context_length",
        "embedding_dim",
        "num_layers",
        "num_heads",
    ],
)
def test_integer_dimensions_must_be_positive(field: str) -> None:
    values = {
        "vocab_size": 32_000,
        "context_length": 2_048,
        "embedding_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
    }
    values[field] = 0

    with pytest.raises(ValueError, match="greater than zero"):
        SLMConfig(**values)


def test_integer_dimensions_reject_boolean_values() -> None:
    with pytest.raises(TypeError, match="num_layers must be an integer"):
        SLMConfig(
            vocab_size=32_000,
            context_length=2_048,
            embedding_dim=512,
            num_layers=True,
            num_heads=8,
        )


def test_embedding_dimension_must_match_attention_heads() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        SLMConfig(
            vocab_size=32_000,
            context_length=2_048,
            embedding_dim=510,
            num_layers=8,
            num_heads=8,
        )


def test_dropout_must_be_in_half_open_unit_interval() -> None:
    base = dict(
        vocab_size=32_000,
        context_length=2_048,
        embedding_dim=512,
        num_layers=8,
        num_heads=8,
    )

    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        SLMConfig(**base, dropout=1.0)

    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        SLMConfig(**base, dropout=-0.1)
