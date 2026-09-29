"""Training experiment metadata and lightweight checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from aria.brain.module import Module
from aria.brain.optim import SGD


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    sequence_length: int
    steps: int
    seed: int | None = None


@dataclass(frozen=True)
class TrainingCheckpoint:
    step: int
    losses: list[float]
    config: TrainingConfig

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TrainingCheckpoint:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            step=int(payload["step"]),
            losses=[float(value) for value in payload["losses"]],
            config=TrainingConfig(**payload["config"]),
        )


def train_with_checkpoint(
    model: Module,
    optimizer: SGD,
    trainer,
    *,
    config: TrainingConfig,
    checkpoint_path: Path,
) -> TrainingCheckpoint:
    """Run training and persist experiment metadata and observed losses."""
    history = trainer.train(config.steps)
    checkpoint = TrainingCheckpoint(
        step=trainer.step_count,
        losses=[item.loss for item in history],
        config=config,
    )
    checkpoint.save(checkpoint_path)
    return checkpoint
