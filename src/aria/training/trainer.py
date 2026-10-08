"""Local training loop for ARIA language models."""

from __future__ import annotations

from dataclasses import dataclass

from aria.brain.language_model import _log_softmax_loss
from aria.brain.module import Module
from aria.brain.optim import LearningRateScheduler, SGD
from aria.evaluation import EvaluationResult, evaluate_language_model
from aria.training.dataset import TokenWindowDataset


@dataclass(frozen=True)
class TrainingStep:
    step: int
    loss: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingEvaluationReport:
    """Train/validation metrics captured before and after a training run."""

    steps: int
    initial_train_loss: float
    final_train_loss: float
    initial_validation: EvaluationResult
    final_validation: EvaluationResult

    @property
    def train_loss_change(self) -> float:
        return self.final_train_loss - self.initial_train_loss

    @property
    def validation_loss_change(self) -> float:
        return self.final_validation.mean_loss - self.initial_validation.mean_loss

    @property
    def final_generalization_gap(self) -> float:
        return self.final_validation.mean_loss - self.final_train_loss


class LanguageModelTrainer:
    """Run deterministic single-example training steps."""

    def __init__(
        self,
        model: Module,
        optimizer: SGD,
        dataset: TokenWindowDataset,
        scheduler: LearningRateScheduler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.scheduler = scheduler
        self.step_count = 0

    def train_step(self, index: int) -> TrainingStep:
        token_ids, targets = self.dataset[index]
        self.optimizer.zero_grad()
        logits = self.model.forward(token_ids)
        loss = _log_softmax_loss(logits, targets)
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        learning_rate = self.optimizer.learning_rate
        if self.scheduler is not None:
            learning_rate = self.scheduler.step(self.optimizer, self.step_count)
        return TrainingStep(
            step=self.step_count,
            loss=loss.item(),
            learning_rate=learning_rate,
        )

    def train(self, steps: int) -> list[TrainingStep]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if len(self.dataset) == 0:
            raise ValueError("dataset must contain at least one training example")
        history = []
        for offset in range(steps):
            dataset_index = (self.step_count + offset) % len(self.dataset)
            history.append(self.train_step(dataset_index))
        return history

    def train_epoch(self) -> list[TrainingStep]:
        """Train exactly once on every dataset example."""

        if len(self.dataset) == 0:
            raise ValueError("dataset must contain at least one training example")
        start = self.step_count % len(self.dataset)
        history = []
        for offset in range(len(self.dataset)):
            history.append(self.train_step((start + offset) % len(self.dataset)))
        return history

    def train_epochs(self, epochs: int) -> list[TrainingStep]:
        """Train for a fixed number of complete dataset passes."""

        if epochs <= 0:
            raise ValueError("epochs must be positive")
        history = []
        for _ in range(epochs):
            history.extend(self.train_epoch())
        return history

    def train_and_evaluate(
        self,
        steps: int,
        validation_dataset: TokenWindowDataset,
    ) -> TrainingEvaluationReport:
        """Train and compare train/validation metrics before and after."""

        if steps <= 0:
            raise ValueError("steps must be positive")
        if len(self.dataset) == 0:
            raise ValueError("dataset must contain at least one training example")
        if len(validation_dataset) == 0:
            raise ValueError("validation dataset must contain at least one example")
        if validation_dataset.sequence_length != self.dataset.sequence_length:
            raise ValueError("train and validation sequence lengths must match")

        train_examples = [self.dataset[index] for index in range(len(self.dataset))]
        validation_examples = [
            validation_dataset[index] for index in range(len(validation_dataset))
        ]

        initial_train = evaluate_language_model(self.model, train_examples)
        initial_validation = evaluate_language_model(
            self.model, validation_examples
        )
        history = self.train(steps)
        final_train = evaluate_language_model(self.model, train_examples)
        final_validation = evaluate_language_model(
            self.model, validation_examples
        )

        return TrainingEvaluationReport(
            steps=len(history),
            initial_train_loss=initial_train.mean_loss,
            final_train_loss=final_train.mean_loss,
            initial_validation=initial_validation,
            final_validation=final_validation,
        )
