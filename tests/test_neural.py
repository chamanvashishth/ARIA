from aria.model import ModelConfig, TinyLanguageModel


def test_model_is_deterministic_for_a_seed() -> None:
    first = TinyLanguageModel(ModelConfig(vocab_size=4, embedding_size=8, seed=11))
    second = TinyLanguageModel(ModelConfig(vocab_size=4, embedding_size=8, seed=11))

    assert first.logits(2) == second.logits(2)


def test_training_reduces_language_model_loss() -> None:
    model = TinyLanguageModel(ModelConfig(vocab_size=3, embedding_size=8, seed=3))
    tokens = [0, 1, 0, 1, 0, 1, 0, 1]

    before = model.loss(tokens)
    history = model.train(tokens, epochs=30, learning_rate=0.1)

    assert history[-1] < before
    assert model.next_token(0) == 1


def test_model_checkpoint_round_trip(tmp_path) -> None:
    model = TinyLanguageModel(ModelConfig(vocab_size=5, embedding_size=4, seed=9))
    model.train([0, 1, 2, 3, 4], epochs=2, learning_rate=0.05)

    path = tmp_path / "aria-model.json"
    model.save(path)
    restored = TinyLanguageModel.load(path)

    assert restored.config == model.config
    assert restored.logits(2) == model.logits(2)


def test_generation_has_requested_length() -> None:
    model = TinyLanguageModel(ModelConfig(vocab_size=4, embedding_size=4))
    generated = model.generate(1, 6)

    assert len(generated) == 6
    assert all(0 <= token < 4 for token in generated)
