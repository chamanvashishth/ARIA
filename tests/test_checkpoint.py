from pathlib import Path

import pytest

from aria.brain import SGD, TransformerLanguageModel
from aria.tokenizer import ByteTokenizer
from aria.brain.parameter import Parameter
from aria.evaluation import evaluate_language_model
from aria.training import (
    LanguageModelTrainer,
    TokenWindowDataset,
    TrainingCheckpoint,
    TrainingConfig,
    restore_training_checkpoint,
    train_with_best_validation_checkpoint,
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


def _parameter_values_by_name(model):
    from aria.training.checkpoint import capture_model_state

    values, _ = capture_model_state(model)
    return values


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


def test_checkpoint_has_version_and_atomic_save(tmp_path: Path) -> None:
    model, optimizer, trainer = _build_training_stack(seed=12)
    path = tmp_path / "checkpoint.json"
    config = TrainingConfig(learning_rate=0.01, sequence_length=4, steps=1, seed=12)

    train_with_checkpoint(
        model,
        optimizer,
        trainer,
        config=config,
        checkpoint_path=path,
    )

    payload = path.read_text(encoding="utf-8")
    assert '"format_version": 1' in payload
    assert not (tmp_path / ".checkpoint.json.tmp").exists()


def test_restore_rejects_sequence_length_mismatch(tmp_path: Path) -> None:
    model, optimizer, trainer = _build_training_stack(seed=13)
    path = tmp_path / "checkpoint.json"
    train_with_checkpoint(
        model,
        optimizer,
        trainer,
        config=TrainingConfig(learning_rate=0.01, sequence_length=4, steps=1, seed=13),
        checkpoint_path=path,
    )

    mismatch_model, mismatch_optimizer, mismatch_trainer = _build_training_stack(seed=13)
    mismatch_trainer.dataset = TokenWindowDataset(
        ByteTokenizer().encode("hello hello hello"),
        sequence_length=2,
        stride=1,
    )
    with pytest.raises(ValueError, match="sequence length does not match"):
        restore_training_checkpoint(
            path,
            mismatch_model,
            mismatch_optimizer,
            mismatch_trainer,
        )


def test_load_rejects_unknown_checkpoint_version(tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    path.write_text(
        '{"format_version": 999, "step": 0, "losses": [], '
        '"config": {"learning_rate": 0.01, "sequence_length": 4, "steps": 1}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported checkpoint format version"):
        TrainingCheckpoint.load(path)


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


def test_best_validation_checkpoint_stops_and_saves_best_state(tmp_path: Path) -> None:
    model, optimizer, trainer = _build_training_stack(seed=21, learning_rate=0.01)
    path = tmp_path / "best.json"

    checkpoint, executed_steps = train_with_best_validation_checkpoint(
        model,
        optimizer,
        trainer,
        validation_dataset=trainer.dataset,
        config=TrainingConfig(
            learning_rate=0.01,
            sequence_length=4,
            steps=6,
            seed=21,
        ),
        checkpoint_path=path,
        patience=2,
        min_delta=100.0,
    )

    assert path.exists()
    assert checkpoint.step == 1
    assert executed_steps == 3
    assert checkpoint.losses == [checkpoint.losses[0]]
    assert checkpoint.parameter_values

    restored_model, restored_optimizer, restored_trainer = _build_training_stack(
        seed=21,
        learning_rate=0.5,
    )
    restored = restore_training_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        restored_trainer,
    )
    assert restored.step == 1
    assert restored_trainer.step_count == 1
    assert restored_optimizer.learning_rate == 0.01
    for name, values in checkpoint.parameter_values.items():
        assert values == pytest.approx(
            _parameter_values_by_name(restored_model)[name],
            rel=1e-12,
            abs=1e-12,
        )


def test_best_validation_checkpoint_rejects_invalid_controls(tmp_path: Path) -> None:
    model, optimizer, trainer = _build_training_stack(seed=22)
    path = tmp_path / "best.json"
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        validation_dataset=trainer.dataset,
        config=TrainingConfig(learning_rate=0.01, sequence_length=4, steps=2),
        checkpoint_path=path,
    )
    with pytest.raises(ValueError, match="patience must be positive"):
        train_with_best_validation_checkpoint(**kwargs, patience=0)
    with pytest.raises(ValueError, match="min_delta must be finite"):
        train_with_best_validation_checkpoint(**kwargs, patience=2, min_delta=-1.0)
