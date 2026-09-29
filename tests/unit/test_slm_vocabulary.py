"""Unit tests for the SLM vocabulary contract."""

import pytest

from aria.slm import Vocabulary


def test_vocabulary_assigns_stable_zero_based_ids() -> None:
    vocabulary = Vocabulary(("<unk>", "hello", "world"))

    assert vocabulary.size == 3
    assert vocabulary.token_to_id("hello") == 1
    assert vocabulary.id_to_token(2) == "world"


def test_vocabulary_is_immutable() -> None:
    vocabulary = Vocabulary(("a", "b"))

    with pytest.raises(AttributeError):
        vocabulary.tokens = ("c",)


def test_vocabulary_rejects_empty_or_duplicate_tokens() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Vocabulary(())

    with pytest.raises(ValueError, match="must be unique"):
        Vocabulary(("a", "a"))


def test_vocabulary_rejects_empty_token_values() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        Vocabulary(("a", ""))


def test_vocabulary_lookup_reports_missing_values() -> None:
    vocabulary = Vocabulary(("a", "b"))

    with pytest.raises(KeyError):
        vocabulary.token_to_id("c")
    with pytest.raises(KeyError):
        vocabulary.id_to_token(2)


def test_vocabulary_rejects_boolean_token_ids() -> None:
    vocabulary = Vocabulary(("a", "b"))

    with pytest.raises(TypeError, match="token_id must be an integer"):
        vocabulary.id_to_token(True)
