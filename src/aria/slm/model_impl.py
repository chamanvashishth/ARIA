"""Dependency-free baseline implementation of the ARIA SLM boundary."""

import math
import random

from .config import SLMConfig
from .model import ModelOutput


class EmbeddingProjectionModel:
    """Small deterministic baseline: token embeddings projected to vocabulary logits.

    This is deliberately not the final transformer architecture. It provides a
    real forward pass behind the model contract without adding a framework
    dependency before the numerical interfaces are settled.
    """

    def __init__(self, config: SLMConfig, seed: int = 0) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.config = config
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(config.embedding_dim)
        self._embeddings = tuple(
            tuple(rng.uniform(-scale, scale) for _ in range(config.embedding_dim))
            for _ in range(config.vocab_size)
        )
        self._projection = tuple(
            tuple(rng.uniform(-scale, scale) for _ in range(config.embedding_dim))
            for _ in range(config.vocab_size)
        )

    def forward(self, input_ids: tuple[tuple[int, ...], ...]) -> ModelOutput:
        """Return vocabulary logits for each input token position."""
        rows: list[tuple[tuple[float, ...], ...]] = []
        for sequence in input_ids:
            if len(sequence) > self.config.context_length:
                raise ValueError("input sequence exceeds context_length")
            positions: list[tuple[float, ...]] = []
            for token_id in sequence:
                self._validate_token_id(token_id)
                embedding = self._embeddings[token_id]
                logits = tuple(
                    sum(value * weight for value, weight in zip(embedding, projection))
                    for projection in self._projection
                )
                positions.append(logits)
            rows.append(tuple(positions))
        return ModelOutput(logits=tuple(rows))

    def _validate_token_id(self, token_id: int) -> None:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token IDs must be integers")
        if not 0 <= token_id < self.config.vocab_size:
            raise ValueError("token ID is outside vocab_size")
