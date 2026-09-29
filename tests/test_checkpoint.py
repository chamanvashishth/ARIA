from pathlib import Path

from aria.brain import SGD, TransformerLanguageModel
from aria.tokenizer import ByteTokenizer
from aria.training import LanguageModelTrainer, TokenWindowDataset
from aria.training.checkpoint import TrainingCheckpoint, TrainingConfig, train_with_checkpoint


def test_training_checkpoint_round_trip(tmp_path: Path) -> None:
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode("hello hello hello")
    dataset = TokenWindowDataset(tokens, sequence_length=4, stride=2)
    model = TransformerLanguageModel(
        vocab_size=tokenizer.VOCAB_SIZE,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        max_sequence_length=4,
        seed=6,
    )
    trainer = LanguageModelTrainer(model, SGD(model.parameters(), learning_rate=0.01), dataset)
    config = TrainingConfig(learning_rate=0.01, sequence_length=4, steps=2, seed=6)
    path = tmp_path / "checkpoint.json"

    checkpoint = train_with_checkpoint(
        model,
        trainer.optimizer,
        trainer,
        config=config,
        checkpoint_path=path,
    )
    restored = TrainingCheckpoint.load(path)

    assert restored.step == checkpoint.step == 2
    assert restored.losses == checkpoint.losses
    assert restored.config == config
