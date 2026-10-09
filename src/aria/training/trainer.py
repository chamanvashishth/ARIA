"""Local training loop for ARIA language models."""

from __future__ import annotations

from dataclasses import dataclass
import math

from aria.brain.language_model import _log_softmax_loss
from aria.brain.module import Module
from aria.brain.optim import LearningRateScheduler, SGD
from aria.evaluation import EvaluationResult, evaluate_language_model
from aria.training.dataset import TokenBatchSampler, TokenWindowDataset


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
    """Run deterministic training steps with model-supported batch execution."""

    def __init__(
        self,
        model: Module,
        optimizer: SGD,
        dataset: TokenWindowDataset,
        scheduler: LearningRateScheduler | None = None,
        batch_size: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.step_count = 0

    def _validate_loss(self, loss: object) -> float:
        value = loss.item()
        if not math.isfinite(value):
            raise ValueError("training loss must be finite")
        return value

    def _validate_gradients(self) -> None:
        for parameter in self.model.parameters():
            if parameter.grad is not None and not all(
                math.isfinite(value) for value in parameter.grad._values
            ):
                raise ValueError("training gradients must be finite")

    def _run_transaction(self, operation):
        """Roll back parameters and trainer/optimizer state if a step fails."""
        parameters = self.model.parameters()
        values_before = [list(parameter._values) for parameter in parameters]
        data_before = [parameter.data for parameter in parameters]
        learning_rate_before = self.optimizer.learning_rate
        step_count_before = self.step_count
        try:
            return operation()
        except Exception:
            for parameter, values, data in zip(parameters, values_before, data_before):
                parameter._values = values
                parameter.data = data
            self.optimizer.zero_grad()
            self.optimizer.learning_rate = learning_rate_before
            self.step_count = step_count_before
            raise

    def train_step(self, index: int) -> TrainingStep:
        return self._run_transaction(lambda: self._train_step(index))

    def _train_step(self, index: int) -> TrainingStep:
        token_ids, targets = self.dataset[index]
        self.optimizer.zero_grad()
        logits = self.model.forward(token_ids)
        loss = _log_softmax_loss(logits, targets)
        loss_value = self._validate_loss(loss)
        loss.backward()
        self._validate_gradients()
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

    def train_batch(self, indices: list[int]) -> TrainingStep:
        """Run one optimizer step over a deterministic mini-batch."""
        return self._run_transaction(lambda: self._train_batch(indices))

    def _train_batch(self, indices: list[int]) -> TrainingStep:
        if not indices:
            raise ValueError("mini-batch must contain at least one example")
        if any(not 0 <= index < len(self.dataset) for index in indices):
            raise IndexError("mini-batch contains an invalid dataset index")

        self.optimizer.zero_grad()
        examples = [self.dataset[index] for index in indices]
        batched_loss = getattr(self.model, "loss_batch", None)
        losses: list[float] = []
        if callable(batched_loss):
            # Models with an explicit batch API can perform one forward/loss
            # pass. TinyLanguageModel flattens equal-length token positions,
            # sharing embedding and vocabulary projections across the batch.
            loss = batched_loss(
                [tokens for tokens, _ in examples],
                [targets for _, targets in examples],
            )
            loss_value = self._validate_loss(loss)
            loss.backward()
            self._validate_gradients()
            losses.append(loss_value)
        else:
            # The Transformer keeps examples separate until attention supports
            # a true [batch, time, hidden] path; this avoids cross-sample leakage.
            for token_ids, targets in examples:
                loss = _log_softmax_loss(self.model.forward(token_ids), targets)
                loss_value = self._validate_loss(loss)
                loss.backward()
                self._validate_gradients()
                losses.append(loss_value)

            # Each example produces a mean sequence loss. Average accumulated
            # gradients so batch size does not change update magnitude.
            scale = 1.0 / len(indices)
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad._values = [value * scale for value in parameter.grad._values]
                    parameter.grad.data = parameter.grad.to_list()

        self.optimizer.step()
        self.step_count += 1
        learning_rate = self.optimizer.learning_rate
        if self.scheduler is not None:
            learning_rate = self.scheduler.step(self.optimizer, self.step_count)
        return TrainingStep(
            step=self.step_count,
            loss=sum(losses) / len(losses),
            learning_rate=learning_rate,
        )

    def train(self, steps: int) -> list[TrainingStep]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if len(self.dataset) == 0:
            raise ValueError("dataset must contain at least one training example")
        history = []
        effective_batch_size = min(self.batch_size, len(self.dataset))
        for _ in range(steps):
            start = (self.step_count * effective_batch_size) % len(self.dataset)
            indices = [
                (start + item) % len(self.dataset)
                for item in range(effective_batch_size)
            ]
            history.append(self.train_batch(indices))
        return history

    def train_epoch(
        self,
        *,
        epoch: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> list[TrainingStep]:
        """Train one dataset pass using deterministic sampled mini-batches."""

        if len(self.dataset) == 0:
            raise ValueError("dataset must contain at least one training example")
        sampler = TokenBatchSampler(
            self.dataset,
            self.batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        history = [
            self.train_batch(indices)
            for indices in sampler.batches(epoch=epoch)
        ]
        return history

    def train_epochs(
        self,
        epochs: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        start_epoch: int = 0,
    ) -> list[TrainingStep]:
        """Train for complete dataset passes with reproducible epoch shuffling."""

        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")
        history = []
        for epoch in range(start_epoch, start_epoch + epochs):
            history.extend(
                self.train_epoch(
                    epoch=epoch,
                    shuffle=shuffle,
                    seed=seed,
                    drop_last=drop_last,
                )
            )
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
