"""Unit tests for the SLM dataset contracts."""

import pytest

from aria.slm import Dataset, TrainingExample


class DummyDataset:
    def __init__(self) -> None:
        self.examples = (TrainingExample((1, 2), (2, 3)),)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


def test_training_example_validates_aligned_token_sequences() -> None:
    example = TrainingExample((1, 2, 3), (2, 3, 4))

    assert example.input_ids == (1, 2, 3)
    assert example.target_ids == (2, 3, 4)


def test_training_example_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TrainingExample((), ())


def test_training_example_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        TrainingExample((1, 2), (2,))


def test_training_example_rejects_invalid_token_ids() -> None:
    with pytest.raises(TypeError, match="only integers"):
        TrainingExample((1, True), (2, 3))

    with pytest.raises(ValueError, match="non-negative IDs"):
        TrainingExample((1, -1), (2, 3))


def test_dataset_contract_accepts_indexed_implementation() -> None:
    dataset = DummyDataset()

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 1
    assert dataset[0] == TrainingExample((1, 2), (2, 3))
