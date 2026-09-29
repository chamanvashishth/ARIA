"""Deterministic token-window dataset utilities."""

from __future__ import annotations


class TokenWindowDataset:
    """Create overlapping next-token training examples from token IDs."""

    def __init__(self, token_ids: list[int], sequence_length: int, stride: int | None = None) -> None:
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
