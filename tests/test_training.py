from aria.brain import SGD, TransformerLanguageModel
from aria.tokenizer import ByteTokenizer
from aria.training import LanguageModelTrainer, TokenWindowDataset


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
