from pathlib import Path

import pytest

from aria.brain import SGD, TransformerLanguageModel
from aria.tokenizer import ByteTokenizer
from aria.training import (
    LanguageModelTrainer,
    TokenWindowDataset,
    TrainingCheckpoint,
    TrainingConfig,
    restore_training_checkpoint,
    train_with_checkpoint,
)


def _build_training_stack(seed: int, learning_rate: float = 0.01):
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode("hello hello hello")
    dataset = TokenWindowDataset(tokens, sequence_length=4, stride=2)
    model = TransformerLanguageModel(
        vocab_size=tokenizer.VOCAB_SIZE,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        max_sequence_length=4,
        seed=seed,
    )
    optimizer = SGD(model.parameters(), learning_rate=learning_rate)
    trainer = LanguageModelTrainer(model, optimizer, dataset)
    return model, optimizer, trainer


def test_training_checkpoint_round_trip(tmp_path: Path) -> None:
    model, optimizer, trainer = _build_training_stack(seed=6)
    config = TrainingConfig(learning_rate=0.01, sequence_length=4, steps=2, seed=6)
    path = tmp_path / "checkpoint.json"

    checkpoint = train_with_checkpoint(
        model,
        optimizer,
        trainer,
        config=config,
        checkpoint_path=path,
    )
    restored = TrainingCheckpoint.load(path)

    assert restored.step == checkpoint.step == 2
    assert restored.losses == checkpoint.losses
    assert restored.config == config
    assert restored.optimizer_learning_rate == 0.01
    assert restored.parameter_values
    assert set(restored.parameter_values) == set(restored.parameter_shapes)


def test_checkpoint_restores_exact_model_output_and_training_continuation(tmp_path: Path) -> None:
    model, optimizer, trainer = _build_training_stack(seed=9, learning_rate=0.02)
    config = TrainingConfig(learning_rate=0.02, sequence_length=4, steps=3, seed=9)
    path = tmp_path / "checkpoint.json"

    train_with_checkpoint(
        model,
        optimizer,
        trainer,
        config=config,
        checkpoint_path=path,
    )

    continuation_input = [1, 2, 3, 4]
    expected_logits = model.forward(continuation_input)._values[:]
    trainer.train(2)
    expected_parameters = [parameter._values[:] for parameter in model.parameters()]

    restored_model, restored_optimizer, restored_trainer = _build_training_stack(
        seed=9,
        learning_rate=999.0,
    )
    checkpoint = restore_training_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        restored_trainer,
    )

    assert checkpoint.step == 3
    assert restored_trainer.step_count == 3
    assert restored_optimizer.learning_rate == 0.02
    assert restored_model.forward(continuation_input)._values == pytest.approx(expected_logits)

    restored_trainer.train(2)
    restored_parameters = [parameter._values[:] for parameter in restored_model.parameters()]

    assert restored_trainer.step_count == 5
    for actual, expected in zip(restored_parameters, expected_parameters):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_restore_rejects_metadata_only_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "metadata-only.json"
    TrainingCheckpoint(
        step=2,
        losses=[1.0, 0.9],
        config=TrainingConfig(learning_rate=0.01, sequence_length=4, steps=2),
    ).save(path)

    model, optimizer, trainer = _build_training_stack(seed=3)
    with pytest.raises(ValueError, match="does not contain model weights"):
        restore_training_checkpoint(path, model, optimizer, trainer)
