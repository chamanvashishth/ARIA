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
    save_training_checkpoint,
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
    assert '"format_version": 2' in payload
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



def test_checkpoint_restores_scheduler_configuration_for_exact_continuation(tmp_path: Path) -> None:
    from aria.brain.optim import ExponentialDecay

    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode("hello hello hello")
    dataset = TokenWindowDataset(tokens, sequence_length=4, stride=2)

    def build(gamma: float, minimum: float):
        model = TransformerLanguageModel(
            vocab_size=tokenizer.VOCAB_SIZE,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            max_sequence_length=4,
            seed=41,
        )
        optimizer = SGD(model.parameters(), learning_rate=0.02)
        scheduler = ExponentialDecay(gamma=gamma, minimum=minimum)
        trainer = LanguageModelTrainer(
            model, optimizer, dataset, scheduler=scheduler
        )
        return model, optimizer, trainer, scheduler

    path = tmp_path / "scheduler-checkpoint.json"
    model, optimizer, trainer, scheduler = build(0.8, 0.001)
    config = TrainingConfig(
        learning_rate=0.02, sequence_length=4, steps=2, seed=41
    )
    train_with_checkpoint(
        model, optimizer, trainer, config=config, checkpoint_path=path
    )
    assert scheduler.gamma == 0.8
    assert optimizer.learning_rate == pytest.approx(0.02 * 0.8**2)

    trainer.train(3)
    expected_parameters = [p._values[:] for p in model.parameters()]
    expected_lr = optimizer.learning_rate

    resumed_model, resumed_optimizer, resumed_trainer, resumed_scheduler = build(
        0.5, 0.02
    )
    checkpoint = restore_training_checkpoint(
        path, resumed_model, resumed_optimizer, resumed_trainer
    )
    assert checkpoint.scheduler_state == {
        "type": "ExponentialDecay",
        "gamma": 0.8,
        "minimum": 0.001,
    }
    assert resumed_scheduler.gamma == 0.8
    assert resumed_scheduler.minimum == 0.001
    resumed_trainer.train(3)

    assert resumed_optimizer.learning_rate == pytest.approx(expected_lr)
    for actual, expected in zip(resumed_model.parameters(), expected_parameters):
        assert actual._values == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_checkpoint_rejects_scheduler_mismatch_without_mutating_model(tmp_path: Path) -> None:
    from aria.brain.optim import ExponentialDecay

    model, optimizer, trainer = _build_training_stack(seed=43, learning_rate=0.01)
    trainer.scheduler = ExponentialDecay(gamma=0.9)
    path = tmp_path / "scheduler-mismatch.json"
    train_with_checkpoint(
        model,
        optimizer,
        trainer,
        config=TrainingConfig(learning_rate=0.01, sequence_length=4, steps=1, seed=43),
        checkpoint_path=path,
    )

    other_model, other_optimizer, other_trainer = _build_training_stack(
        seed=43, learning_rate=0.01
    )
    before = [parameter._values[:] for parameter in other_model.parameters()]
    with pytest.raises(ValueError, match="trainer has no scheduler"):
        restore_training_checkpoint(path, other_model, other_optimizer, other_trainer)
    assert [parameter._values for parameter in other_model.parameters()] == before



def test_restore_rejects_non_finite_parameter_without_partial_mutation(tmp_path: Path) -> None:
    import json

    model, optimizer, trainer = _build_training_stack(seed=51, learning_rate=0.01)
    path = tmp_path / "corrupted.json"
    train_with_checkpoint(
        model,
        optimizer,
        trainer,
        config=TrainingConfig(learning_rate=0.01, sequence_length=4, steps=1, seed=51),
        checkpoint_path=path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    first_name = next(iter(payload["parameter_values"]))
    payload["parameter_values"][first_name][0] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")

    target_model, target_optimizer, target_trainer = _build_training_stack(
        seed=52, learning_rate=0.03
    )
    before = [parameter._values[:] for parameter in target_model.parameters()]
    before_lr = target_optimizer.learning_rate
    before_step = target_trainer.step_count
    with pytest.raises(ValueError, match="parameter values must be finite"):
        restore_training_checkpoint(path, target_model, target_optimizer, target_trainer)

    assert [parameter._values for parameter in target_model.parameters()] == before
    assert target_optimizer.learning_rate == before_lr
    assert target_trainer.step_count == before_step


def test_restore_model_state_validates_all_parameters_before_mutating() -> None:
    from aria.training.checkpoint import capture_model_state, restore_model_state

    model, _, _ = _build_training_stack(seed=53)
    values, shapes = capture_model_state(model)
    names = list(values)
    values[names[0]] = [value + 1.0 for value in values[names[0]]]
    shapes[names[-1]] = [999]
    before = [parameter._values[:] for parameter in model.parameters()]

    with pytest.raises(ValueError, match="checkpoint shape mismatch"):
        restore_model_state(model, values, shapes)

    assert [parameter._values for parameter in model.parameters()] == before



def test_save_training_checkpoint_snapshots_without_training(tmp_path: Path) -> None:
    model, optimizer, trainer = _build_training_stack(seed=61)
    path = tmp_path / "snapshot.json"
    before = [parameter._values[:] for parameter in model.parameters()]

    checkpoint = save_training_checkpoint(
        model,
        optimizer,
        trainer,
        config=TrainingConfig(learning_rate=0.01, sequence_length=4, steps=5, seed=61),
        checkpoint_path=path,
        losses=[1.25],
    )

    assert path.exists()
    assert trainer.step_count == 0
    assert checkpoint.step == 0
    assert checkpoint.losses == [1.25]
    for actual, expected in zip(
        [parameter._values for parameter in model.parameters()], before
    ):
        assert actual == expected


def test_save_failure_cleans_temporary_file_and_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text("previous valid checkpoint", encoding="utf-8")
    checkpoint = TrainingCheckpoint(
        step=0,
        losses=[],
        config=TrainingConfig(learning_rate=0.01, sequence_length=4, steps=1),
    )

    def fail_replace(self, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        checkpoint.save(path)

    assert path.read_text(encoding="utf-8") == "previous valid checkpoint"
    assert not (tmp_path / ".checkpoint.json.tmp").exists()


@pytest.mark.parametrize("version", [0, 1])
def test_legacy_metadata_checkpoint_remains_loadable(tmp_path: Path, version: int) -> None:
    path = tmp_path / f"legacy-v{version}.json"
    path.write_text(
        (
            '{"step": 2, "losses": [1.0], '
            '"config": {"learning_rate": 0.01, "sequence_length": 4, "steps": 2}}'
            if version == 0
            else '{"format_version": 1, "step": 2, "losses": [1.0], '
            '"config": {"learning_rate": 0.01, "sequence_length": 4, "steps": 2}}'
        ),
        encoding="utf-8",
    )

    checkpoint = TrainingCheckpoint.load(path)
    assert checkpoint.step == 2
    assert checkpoint.losses == [1.0]
    assert checkpoint.parameter_values == {}
    assert checkpoint.scheduler_state is None
