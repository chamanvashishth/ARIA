import pytest

from aria.brain import SGD, TinyLanguageModel, TransformerLanguageModel
from aria.evaluation import evaluate_language_model
from aria.tokenizer import ByteTokenizer
from aria.training import (
    LanguageModelTrainer,
    TokenWindowDataset,
    build_train_validation_datasets,
    split_token_ids,
)


def test_token_window_dataset_creates_next_token_pairs() -> None:
    dataset = TokenWindowDataset([1, 2, 3, 4, 5], sequence_length=3)
    assert len(dataset) == 1
    assert dataset[0] == ([1, 2, 3], [2, 3, 4])


def test_trainer_runs_and_updates_model() -> None:
    tokens = ByteTokenizer().encode("hello hello hello")
    dataset = TokenWindowDataset(tokens, sequence_length=4, stride=2)
    model = TransformerLanguageModel(
        vocab_size=ByteTokenizer.VOCAB_SIZE,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        max_sequence_length=4,
        seed=5,
    )
    optimizer = SGD(model.parameters(), learning_rate=0.01)
    before = list(model.lm_head.weight._values)

    trainer = LanguageModelTrainer(model, optimizer, dataset)
    history = trainer.train(2)

    assert len(history) == 2
    assert all(step.loss > 0 for step in history)
    assert model.lm_head.weight._values != before
    assert trainer.step_count == 2


def test_tiny_corpus_training_reduces_loss() -> None:
    # A deliberately simple repeating pattern checks the learning loop, not
    # general language quality or held-out generalization.
    token_ids = [1, 2] * 16
    dataset = TokenWindowDataset(token_ids, sequence_length=2, stride=1)
    model = TinyLanguageModel(vocab_size=3, embedding_dim=8, seed=23)
    optimizer = SGD(model.parameters(), learning_rate=0.05)
    examples = [dataset[index] for index in range(len(dataset))]

    initial = evaluate_language_model(model, examples).mean_loss
    trainer = LanguageModelTrainer(model, optimizer, dataset)
    trainer.train(120)
    final = evaluate_language_model(model, examples).mean_loss

    assert final < initial * 0.8
    assert trainer.step_count == 120


def test_split_token_ids_is_deterministic_and_non_overlapping() -> None:
    train, validation = split_token_ids(list(range(10)), validation_fraction=0.3)
    assert train == list(range(7))
    assert validation == [7, 8, 9]
    assert set(train).isdisjoint(validation)


def test_build_train_validation_datasets_preserves_split_boundary() -> None:
    train, validation = build_train_validation_datasets(
        list(range(16)),
        sequence_length=3,
        stride=1,
        validation_fraction=0.25,
    )

    assert train.token_ids == list(range(12))
    assert validation.token_ids == [12, 13, 14, 15]
    assert train[-1] == ([8, 9, 10], [9, 10, 11])
    assert validation[0] == ([12, 13, 14], [13, 14, 15])


def test_train_and_evaluate_reports_train_and_validation_changes() -> None:
    train, validation = build_train_validation_datasets(
        [1, 2] * 20,
        sequence_length=2,
        stride=1,
        validation_fraction=0.25,
    )
    model = TinyLanguageModel(vocab_size=3, embedding_dim=8, seed=23)
    optimizer = SGD(model.parameters(), learning_rate=0.05)
    trainer = LanguageModelTrainer(model, optimizer, train)

    report = trainer.train_and_evaluate(60, validation)

    assert report.steps == 60
    assert report.final_train_loss < report.initial_train_loss
    assert report.final_validation.token_count == len(validation) * validation.sequence_length
    assert report.final_generalization_gap == pytest.approx(
        report.final_validation.mean_loss - report.final_train_loss
    )


def test_train_and_evaluate_rejects_mismatched_sequence_lengths() -> None:
    train = TokenWindowDataset([1, 2, 1, 2, 1, 2], sequence_length=2)
    validation = TokenWindowDataset([1, 2, 1, 2, 1], sequence_length=3)
    model = TinyLanguageModel(vocab_size=3, embedding_dim=4, seed=5)
    trainer = LanguageModelTrainer(model, SGD(model.parameters(), learning_rate=0.01), train)

    with pytest.raises(ValueError, match="sequence lengths must match"):
        trainer.train_and_evaluate(1, validation)


def test_trainer_epoch_visits_every_dataset_example_once() -> None:
    dataset = TokenWindowDataset([1, 2, 3, 4, 5, 6], sequence_length=2, stride=2)
    model = TinyLanguageModel(vocab_size=7, embedding_dim=4, seed=11)
    trainer = LanguageModelTrainer(model, SGD(model.parameters(), learning_rate=0.01), dataset)

    history = trainer.train_epoch()

    assert len(history) == len(dataset)
    assert [item.step for item in history] == [1, 2]
    assert trainer.step_count == len(dataset)


def test_scheduler_changes_learning_rate_deterministically() -> None:
    from aria.brain import StepDecay

    model = TinyLanguageModel(vocab_size=7, embedding_dim=4, seed=12)
    optimizer = SGD(model.parameters(), learning_rate=0.1)
    trainer = LanguageModelTrainer(
        model,
        optimizer,
        TokenWindowDataset([1, 2, 3, 4, 5, 6], sequence_length=2, stride=2),
        scheduler=StepDecay(drop_every=2, gamma=0.5),
    )

    history = trainer.train(4)

    assert [round(item.learning_rate, 8) for item in history] == [0.1, 0.05, 0.05, 0.025]
    assert optimizer.learning_rate == pytest.approx(0.025)


def test_scheduler_rejects_invalid_configuration() -> None:
    from aria.brain import ExponentialDecay, StepDecay

    with pytest.raises(ValueError, match="gamma"):
        ExponentialDecay(gamma=0)
    with pytest.raises(ValueError, match="drop_every"):
        StepDecay(drop_every=0)


def test_mini_batch_averages_gradients_and_counts_one_optimizer_step() -> None:
    dataset = TokenWindowDataset([1, 2, 1, 2, 1, 2], sequence_length=2, stride=1)
    model = TinyLanguageModel(vocab_size=3, embedding_dim=4, seed=31)
    optimizer = SGD(model.parameters(), learning_rate=0.01)
    trainer = LanguageModelTrainer(model, optimizer, dataset, batch_size=2)

    history = trainer.train(1)

    assert len(history) == 1
    assert history[0].step == 1
    assert history[0].loss > 0
    assert trainer.step_count == 1


def test_mini_batch_epoch_uses_every_example_once() -> None:
    dataset = TokenWindowDataset([1, 2, 3, 4, 5, 6, 1], sequence_length=2, stride=1)
    model = TinyLanguageModel(vocab_size=7, embedding_dim=4, seed=32)
    trainer = LanguageModelTrainer(
        model,
        SGD(model.parameters(), learning_rate=0.01),
        dataset,
        batch_size=2,
    )

    history = trainer.train_epoch()

    assert len(history) == 3
    assert [item.step for item in history] == [1, 2, 3]
    assert trainer.step_count == 3


def test_batch_size_must_be_positive() -> None:
    model = TinyLanguageModel(vocab_size=3, embedding_dim=4, seed=33)
    dataset = TokenWindowDataset([1, 2, 1, 2], sequence_length=2)
    with pytest.raises(ValueError, match="batch_size"):
        LanguageModelTrainer(model, SGD(model.parameters(), learning_rate=0.01), dataset, batch_size=0)
