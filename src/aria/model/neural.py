"""Small, fully local neural language-model foundation for ARIA.

This is intentionally tiny: embeddings -> linear vocabulary head. It is a real
trainable neural language model, not a prompt wrapper. The implementation uses
only the Python standard library so the learning loop is easy to inspect and
test before introducing larger tensor backends.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    embedding_size: int = 32
    seed: int = 7

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.embedding_size < 1:
            raise ValueError("embedding_size must be positive")


class TinyLanguageModel:
    """A real trainable next-token model owned by ARIA.

    It predicts the next token from the current token. This is a deliberately
    small learning core, not the final ARIA SLM architecture.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        rng = random.Random(config.seed)
        scale = 1.0 / math.sqrt(config.embedding_size)
        self.embeddings = [
            [rng.uniform(-scale, scale) for _ in range(config.embedding_size)]
            for _ in range(config.vocab_size)
        ]
        self.output = [
            [rng.uniform(-scale, scale) for _ in range(config.vocab_size)]
            for _ in range(config.embedding_size)
        ]
        self.bias = [0.0] * config.vocab_size

    def logits(self, token_id: int) -> list[float]:
        self._check_token(token_id)
        hidden = self.embeddings[token_id]
        return [
            sum(hidden[i] * self.output[i][target] for i in range(self.config.embedding_size))
            + self.bias[target]
            for target in range(self.config.vocab_size)
        ]

    def probabilities(self, token_id: int) -> list[float]:
        values = self.logits(token_id)
        maximum = max(values)
        exponentials = [math.exp(value - maximum) for value in values]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def loss(self, token_ids: Sequence[int]) -> float:
        if len(token_ids) < 2:
            raise ValueError("training sequence must contain at least two tokens")
        total = 0.0
        for current, target in zip(token_ids, token_ids[1:]):
            probability = self.probabilities(current)[target]
            total -= math.log(max(probability, 1e-12))
        return total / (len(token_ids) - 1)

    def train_step(self, current: int, target: int, learning_rate: float) -> float:
        self._check_token(current)
        self._check_token(target)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        hidden = self.embeddings[current]
        logits = self.logits(current)
        maximum = max(logits)
        exp_values = [math.exp(value - maximum) for value in logits]
        total = sum(exp_values)
        probabilities = [value / total for value in exp_values]
        loss = -math.log(max(probabilities[target], 1e-12))

        grad_logits = probabilities
        grad_logits[target] -= 1.0

        grad_hidden = [0.0] * self.config.embedding_size
        for i in range(self.config.embedding_size):
            grad_hidden[i] = sum(
                self.output[i][vocab_id] * grad_logits[vocab_id]
                for vocab_id in range(self.config.vocab_size)
            )

        for i in range(self.config.embedding_size):
            hidden_value = hidden[i]
            for vocab_id in range(self.config.vocab_size):
                self.output[i][vocab_id] -= learning_rate * hidden_value * grad_logits[vocab_id]
            self.embeddings[current][i] -= learning_rate * grad_hidden[i]

        for vocab_id in range(self.config.vocab_size):
            self.bias[vocab_id] -= learning_rate * grad_logits[vocab_id]

        return loss

    def train(self, token_ids: Sequence[int], epochs: int = 1, learning_rate: float = 0.05) -> list[float]:
        if len(token_ids) < 2:
            raise ValueError("training sequence must contain at least two tokens")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        history = []
        for _ in range(epochs):
            for current, target in zip(token_ids, token_ids[1:]):
                self.train_step(current, target, learning_rate)
            history.append(self.loss(token_ids))
        return history

    def next_token(self, token_id: int) -> int:
        probabilities = self.probabilities(token_id)
        return max(range(self.config.vocab_size), key=probabilities.__getitem__)

    def generate(self, seed_token: int, length: int) -> list[int]:
        if length < 1:
            raise ValueError("length must be positive")
        generated = [seed_token]
        current = seed_token
        for _ in range(length - 1):
            current = self.next_token(current)
            generated.append(current)
        return generated

    def save(self, path: str | Path) -> None:
        payload = {
            "config": {
                "vocab_size": self.config.vocab_size,
                "embedding_size": self.config.embedding_size,
                "seed": self.config.seed,
            },
            "embeddings": self.embeddings,
            "output": self.output,
            "bias": self.bias,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TinyLanguageModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(ModelConfig(**payload["config"]))
        model.embeddings = payload["embeddings"]
        model.output = payload["output"]
        model.bias = payload["bias"]
        return model

    def _check_token(self, token_id: int) -> None:
        if not 0 <= token_id < self.config.vocab_size:
            raise ValueError(f"token id out of range: {token_id}")
