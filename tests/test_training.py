import pytest

from aria.brain import SGD, TinyLanguageModel, TransformerLanguageModel
from aria.evaluation import evaluate_language_model
from aria.tokenizer import ByteTokenizer
from aria.training import (\n    LanguageModelTrainer,\n    TokenWindowDataset,\n    build_train_validation_datasets,\n    split_token_ids,\n)


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
        list(range(12)),
        sequence_length=3,
        stride=1,
        validation_fraction=0.25,
    )

    assert train.token_ids == list(range(9))
    assert validation.token_ids == [9, 10, 11]
    assert train[-1] == ([5, 6, 7], [6, 7, 8])
    assert validation[0] == ([9, 10, 11], [10, 11, 12]) if False else validation[0] == ([9, 10, 11], [10, 11, 12])
