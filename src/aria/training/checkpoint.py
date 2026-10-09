"""Serializable training checkpoints for ARIA model weights and progress."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aria.brain.module import Module
from aria.brain.optim import ExponentialDecay, SGD, StepDecay
from aria.brain.parameter import Parameter
from aria.evaluation import evaluate_language_model


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    sequence_length: int
    steps: int
    seed: int | None = None
    batch_size: int = 1


@dataclass(frozen=True)
class TrainingCheckpoint:
    """A JSON checkpoint containing model weights and resumable training state.

    parameter_values maps stable module paths to flattened parameter values.
    Older metadata-only checkpoints remain loadable with an empty mapping.
    """

    FORMAT_VERSION = 2

    step: int
    losses: list[float]
    config: TrainingConfig
    parameter_values: dict[str, list[float]] = field(default_factory=dict)
    parameter_shapes: dict[str, list[int]] = field(default_factory=dict)
    optimizer_learning_rate: float | None = None
    scheduler_state: dict[str, Any] | None = None

    def save(self, path: Path) -> None:
        if self.step < 0:
            raise ValueError("checkpoint step cannot be negative")
        if any(not math.isfinite(loss) for loss in self.losses):
            raise ValueError("checkpoint losses must be finite")
        if self.config.learning_rate <= 0:
            raise ValueError("checkpoint learning rate must be positive")
        if self.config.sequence_length <= 0 or self.config.steps <= 0 or self.config.batch_size <= 0:
            raise ValueError("checkpoint training dimensions must be positive")
        if self.optimizer_learning_rate is not None and self.optimizer_learning_rate <= 0:
            raise ValueError("checkpoint optimizer learning rate must be positive")

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["format_version"] = self.FORMAT_VERSION
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "TrainingCheckpoint":
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("format_version", 0)
        if version not in (0, 1, cls.FORMAT_VERSION):
            raise ValueError(f"unsupported checkpoint format version: {version}")
        checkpoint = cls(
            step=int(payload["step"]),
            losses=[float(value) for value in payload["losses"]],
            config=TrainingConfig(**payload["config"]),
            parameter_values={
                str(name): [float(value) for value in values]
                for name, values in payload.get("parameter_values", {}).items()
            },
            parameter_shapes={
                str(name): [int(value) for value in shape]
                for name, shape in payload.get("parameter_shapes", {}).items()
            },
            optimizer_learning_rate=(
                None
                if payload.get("optimizer_learning_rate") is None
                else float(payload["optimizer_learning_rate"])
            ),
            scheduler_state=payload.get("scheduler_state"),
        )
        if checkpoint.step < 0:
            raise ValueError("checkpoint step cannot be negative")
        if any(not math.isfinite(loss) for loss in checkpoint.losses):
            raise ValueError("checkpoint losses must be finite")
        if checkpoint.config.learning_rate <= 0:
            raise ValueError("checkpoint learning rate must be positive")
        if checkpoint.config.sequence_length <= 0 or checkpoint.config.steps <= 0 or checkpoint.config.batch_size <= 0:
            raise ValueError("checkpoint training dimensions must be positive")
        if (
            checkpoint.optimizer_learning_rate is not None
            and checkpoint.optimizer_learning_rate <= 0
        ):
            raise ValueError("checkpoint optimizer learning rate must be positive")
        return checkpoint


def _named_parameters(module: Module) -> dict[str, Parameter]:
    """Collect parameters by deterministic module/attribute paths."""

    result: dict[str, Parameter] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Parameter):
            if path in result:
                raise ValueError(f"duplicate parameter path: {path}")
            result[path] = value
            return
        if isinstance(value, Module):
            for name, child in value.__dict__.items():
                child_path = f"{path}.{name}" if path else name
                visit(child, child_path)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}.{index}")

    visit(module, "")
    return result


def capture_model_state(model: Module) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    """Return JSON-safe model parameter values and shapes."""

    named = _named_parameters(model)
    return (
        {name: list(parameter._values) for name, parameter in named.items()},
        {name: list(parameter.shape) for name, parameter in named.items()},
    )


def restore_model_state(
    model: Module,
    parameter_values: dict[str, list[float]],
    parameter_shapes: dict[str, list[int]],
) -> None:
    """Restore model weights after validating names, shapes, and value counts."""

    named = _named_parameters(model)
    expected_names = set(named)
    actual_names = set(parameter_values)
    if expected_names != actual_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            "checkpoint parameter names do not match "
            f"(missing={missing}, unexpected={unexpected})"
        )

    for name, parameter in named.items():
        values = parameter_values[name]
        shape = tuple(parameter_shapes.get(name, []))
        if shape != parameter.shape:
            raise ValueError(
                f"checkpoint shape mismatch for {name}: expected {parameter.shape}, got {shape}"
            )
        if len(values) != len(parameter._values):
            raise ValueError(
                f"checkpoint value count mismatch for {name}: "
                f"expected {len(parameter._values)}, got {len(values)}"
            )
        parameter._values = list(values)
        parameter.data = _reshape(parameter._values, parameter.shape)
        parameter.zero_grad()


def train_with_checkpoint(
    model: Module,
    optimizer: SGD,
    trainer,
    *,
    config: TrainingConfig,
    checkpoint_path: Path,
) -> TrainingCheckpoint:
    """Run training and persist model weights plus observed losses."""

    if trainer.batch_size != config.batch_size:
        raise ValueError("checkpoint batch size does not match trainer")

    history = trainer.train(config.steps)
    parameter_values, parameter_shapes = capture_model_state(model)
    checkpoint = TrainingCheckpoint(
        step=trainer.step_count,
        losses=[item.loss for item in history],
        config=config,
        parameter_values=parameter_values,
        parameter_shapes=parameter_shapes,
        optimizer_learning_rate=optimizer.learning_rate,
        scheduler_state=_capture_scheduler_state(trainer.scheduler),
    )
    checkpoint.save(checkpoint_path)
    return checkpoint



def train_with_best_validation_checkpoint(
    model: Module,
    optimizer: SGD,
    trainer,
    *,
    validation_dataset,
    config: TrainingConfig,
    checkpoint_path: Path,
    patience: int = 5,
    min_delta: float = 0.0,
) -> tuple[TrainingCheckpoint, int]:
    """Train with validation checks and persist the best validation model.

    Returns the best checkpoint and the number of executed training steps.
    The saved checkpoint contains the model state at the best validation loss,
    not necessarily the final training step.
    """
    if patience <= 0:
        raise ValueError("patience must be positive")
    if config.steps <= 0 or config.learning_rate <= 0 or config.batch_size <= 0:
        raise ValueError("checkpoint training configuration must be positive")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative")
    if len(trainer.dataset) == 0:
        raise ValueError("training dataset must contain at least one example")
    if validation_dataset.sequence_length != trainer.dataset.sequence_length:
        raise ValueError("train and validation sequence lengths must match")
    if config.sequence_length != trainer.dataset.sequence_length:
        raise ValueError("config sequence length does not match trainer dataset")
    if len(validation_dataset) == 0:
        raise ValueError("validation dataset must contain at least one example")

    validation_examples = [
        validation_dataset[index] for index in range(len(validation_dataset))
    ]
    best_loss = math.inf
    stale_steps = 0
    losses: list[float] = []
    best_checkpoint: TrainingCheckpoint | None = None

    if trainer.batch_size != config.batch_size:
        trainer.batch_size = config.batch_size

    for _ in range(config.steps):
        effective_batch_size = min(trainer.batch_size, len(trainer.dataset))
        start = (trainer.step_count * effective_batch_size) % len(trainer.dataset)
        indices = [
            (start + item) % len(trainer.dataset)
            for item in range(effective_batch_size)
        ]
        step = trainer.train_batch(indices)
        losses.append(step.loss)
        validation = evaluate_language_model(model, validation_examples)
        if validation.mean_loss < best_loss - min_delta:
            best_loss = validation.mean_loss
            stale_steps = 0
            parameter_values, parameter_shapes = capture_model_state(model)
            best_checkpoint = TrainingCheckpoint(
                step=trainer.step_count,
                losses=list(losses),
                config=config,
                parameter_values=parameter_values,
                parameter_shapes=parameter_shapes,
                optimizer_learning_rate=optimizer.learning_rate,
                scheduler_state=_capture_scheduler_state(trainer.scheduler),
            )
            best_checkpoint.save(checkpoint_path)
        else:
            stale_steps += 1
            if stale_steps >= patience:
                break

    if best_checkpoint is None:
        raise RuntimeError("training produced no validation checkpoint")
    return best_checkpoint, trainer.step_count


def _capture_scheduler_state(scheduler) -> dict[str, Any] | None:
    """Capture configuration for schedulers whose behavior is deterministic."""
    if scheduler is None:
        return None
    if isinstance(scheduler, ExponentialDecay):
        return {
            "type": "ExponentialDecay",
            "gamma": scheduler.gamma,
            "minimum": scheduler.minimum,
        }
    if isinstance(scheduler, StepDecay):
        return {
            "type": "StepDecay",
            "drop_every": scheduler.drop_every,
            "gamma": scheduler.gamma,
            "minimum": scheduler.minimum,
        }
    raise ValueError(
        "checkpointing does not support scheduler type "
        f"{type(scheduler).__name__}; use ExponentialDecay or StepDecay"
    )


def _restore_scheduler_state(scheduler, state: dict[str, Any] | None) -> None:
    """Restore a supported scheduler configuration after validating its type."""
    if state is None:
        return
    if scheduler is None:
        raise ValueError("checkpoint contains scheduler state but trainer has no scheduler")
    scheduler_type = state.get("type")
    if scheduler_type == "ExponentialDecay" and isinstance(scheduler, ExponentialDecay):
        restored = ExponentialDecay(
            gamma=float(state["gamma"]),
            minimum=float(state["minimum"]),
        )
        scheduler.gamma = restored.gamma
        scheduler.minimum = restored.minimum
        return
    if scheduler_type == "StepDecay" and isinstance(scheduler, StepDecay):
        restored = StepDecay(
            drop_every=int(state["drop_every"]),
            gamma=float(state["gamma"]),
            minimum=float(state["minimum"]),
        )
        scheduler.drop_every = restored.drop_every
        scheduler.gamma = restored.gamma
        scheduler.minimum = restored.minimum
        return
    raise ValueError(
        "checkpoint scheduler does not match trainer scheduler "
        f"(checkpoint={scheduler_type}, trainer={type(scheduler).__name__})"
    )


def restore_training_checkpoint(
    checkpoint_path: Path,
    model: Module,
    optimizer: SGD,
    trainer,
) -> TrainingCheckpoint:
    """Restore model weights and training progress into an existing trainer."""

    checkpoint = TrainingCheckpoint.load(checkpoint_path)
    if not checkpoint.parameter_values:
        raise ValueError("checkpoint does not contain model weights")
    if checkpoint.config.sequence_length != trainer.dataset.sequence_length:
        raise ValueError(
            "checkpoint sequence length does not match trainer dataset: "
            f"{checkpoint.config.sequence_length} != {trainer.dataset.sequence_length}"
        )

    _restore_scheduler_state(trainer.scheduler, checkpoint.scheduler_state)
    restore_model_state(model, checkpoint.parameter_values, checkpoint.parameter_shapes)
    if checkpoint.optimizer_learning_rate is not None:
        optimizer.learning_rate = checkpoint.optimizer_learning_rate
    trainer.step_count = checkpoint.step
    trainer.batch_size = checkpoint.config.batch_size
    return checkpoint


def _reshape(values: list[float], shape: tuple[int, ...]) -> object:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values)
    width = 1
    for dim in shape[1:]:
        width *= dim
    return [_reshape(values[i * width:(i + 1) * width], shape[1:]) for i in range(shape[0])]
