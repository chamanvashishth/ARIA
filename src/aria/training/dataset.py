"""Deterministic token-window dataset utilities."""

from __future__ import annotations


class TokenWindowDataset:
    """Create overlapping next-token training examples from token IDs."""

    def __init__(
        self,
        token_ids: list[int],
        sequence_length: int,
        stride: int | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence length must be positive")
        if len(token_ids) < sequence_length + 1:
            raise ValueError("at least sequence_length + 1 tokens are required")
        self.token_ids = list(token_ids)
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    def __len__(self) -> int:
        return 1 + (len(self.token_ids) - self.sequence_length - 1) // self.stride

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.stride
        inputs = self.token_ids[start:start + self.sequence_length]
        targets = self.token_ids[start + 1:start + self.sequence_length + 1]
        return inputs, targets


def split_token_ids(
    token_ids: list[int],
    *,
    validation_fraction: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Split a token stream into non-overlapping train and validation streams."""

    if len(token_ids) < 4:
        raise ValueError("at least 4 tokens are required for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    validation_size = max(1, int(len(token_ids) * validation_fraction))
    train_size = len(token_ids) - validation_size
    if train_size < 2 or validation_size < 2:
        raise ValueError("train and validation streams must each contain at least 2 tokens")

    return list(token_ids[:train_size]), list(token_ids[train_size:])


def build_train_validation_datasets(
    token_ids: list[int],
    sequence_length: int,
    *,
    stride: int | None = None,
    validation_fraction: float = 0.2,
) -> tuple[TokenWindowDataset, TokenWindowDataset]:
    """Create deterministic, non-overlapping train and validation datasets."""

    train_tokens, validation_tokens = split_token_ids(
        token_ids,
        validation_fraction=validation_fraction,
    )
    return (
        TokenWindowDataset(train_tokens, sequence_length, stride),
        TokenWindowDataset(validation_tokens, sequence_length, stride),
    )


class TokenBatchSampler:
    """Yield deterministic mini-batches of dataset indices.

    Shuffling is reproducible for a given seed and epoch. Unless drop_last is
    enabled, the final batch may be smaller than batch_size.
    """

    def __init__(
        self,
        dataset: TokenWindowDataset,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

    def batches(self, epoch: int = 0) -> list[list[int]]:
        """Return this epoch's batches; epoch is a non-negative counter."""
        import random

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(indices)
        batches = [
            indices[start:start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches
