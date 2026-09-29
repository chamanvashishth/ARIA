"""Unit tests for SLM preprocessing."""

import pytest

from aria.slm import preprocess_text


class DummyTokenizer:
    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(int(char) for char in text.split())


def test_preprocess_text_creates_shifted_training_windows() -> None:
    examples = preprocess_text("1 2 3 4 5", DummyTokenizer(), context_length=2)
    assert len(examples) == 2
    assert examples[0].input_ids == (1, 2)
    assert examples[0].target_ids == (2, 3)
    assert examples[1].input_ids == (3, 4)
    assert examples[1].target_ids == (4, 5)


def test_preprocess_text_discards_incomplete_final_window() -> None:
    examples = preprocess_text("1 2 3 4 5 6", DummyTokenizer(), context_length=4)
    assert len(examples) == 1
    assert examples[0].input_ids == (1, 2, 3, 4)
    assert examples[0].target_ids == (2, 3, 4, 5)


def test_preprocess_text_validates_context_length() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        preprocess_text("1 2", DummyTokenizer(), context_length=0)
    with pytest.raises(TypeError, match="must be an integer"):
        preprocess_text("1 2", DummyTokenizer(), context_length=True)


def test_preprocess_text_validates_tokenizer_output() -> None:
    class InvalidTokenizer:
        def encode(self, text: str) -> tuple[int, ...]:
            return (1, -2)

    with pytest.raises(ValueError, match="non-negative IDs"):
        preprocess_text("ignored", InvalidTokenizer(), context_length=1)
