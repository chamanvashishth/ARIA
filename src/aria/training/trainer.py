"""Local training loop for ARIA language models."""

from __future__ import annotations

from dataclasses import dataclass

from aria.brain.module import Module
from aria.brain.optim import SGD
from aria.brain.tensor import Tensor
from aria.brain.language_model import _log_softmax_loss
from aria.training.dataset import TokenWindowDataset


@dataclass(frozen=True)
class TrainingStep:
    step: int
    loss: float


class LanguageModelTrainer:
    """Run deterministic single-example training steps."""

    def __init__(self, model: Module, optimizer: SGD, dataset: TokenWindowDataset) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.step_count = 0

    def train_step(self, index: int) -> TrainingStep:
        token_ids, targets = self.dataset[index]
        self.optimizer.zero_grad()
        logits = self.model.forward(token_ids)
        loss = _log_softmax_loss(logits, targets)
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        return TrainingStep(step=self.step_count, loss=loss.item())

    def train(self, steps: int) -> list[TrainingStep]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        history = []
        for step in range(steps):
            history.append(self.train_step(step % len(self.dataset)))
        return history
