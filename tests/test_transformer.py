from aria.brain import TransformerLanguageModel


def test_transformer_shapes_and_causality_path() -> None:
    model = TransformerLanguageModel(
        vocab_size=12,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        max_sequence_length=16,
        seed=7,
    )
    logits = model.forward([1, 2, 3, 4])
    assert logits.shape == (4, 12)


def test_transformer_has_trainable_parameters() -> None:
    model = TransformerLanguageModel(
        vocab_size=10,
        hidden_size=6,
        intermediate_size=12,
        num_layers=2,
        max_sequence_length=8,
        seed=8,
    )
    assert len(model.parameters()) > 0
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_transformer_backward_reaches_parameters() -> None:
    model = TransformerLanguageModel(
        vocab_size=9,
        hidden_size=6,
        intermediate_size=12,
        num_layers=1,
        max_sequence_length=8,
        seed=9,
    )
    logits = model.forward([1, 2, 3])
    loss = logits.sum()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_transformer_rejects_long_sequence() -> None:
    model = TransformerLanguageModel(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        max_sequence_length=2,
    )
    try:
        model.forward([1, 2, 3])
    except ValueError as exc:
        assert "context length" in str(exc)
    else:
        raise AssertionError("expected context-length validation")
