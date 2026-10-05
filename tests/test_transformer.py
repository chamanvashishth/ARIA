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

def test_transformer_embedding_gradients_match_finite_differences() -> None:
    from aria.evaluation import check_gradients

    model = TransformerLanguageModel(
        vocab_size=4,
        hidden_size=2,
        intermediate_size=3,
        num_layers=1,
        max_sequence_length=3,
        seed=11,
    )

    report = check_gradients(
        lambda: model.forward([1, 2]).sum(),
        model.parameters(),
        max_checks=4,
    )

    assert report.passed
    assert report.checked_values == 4

def test_rms_norm_gradients_match_finite_differences() -> None:
    from aria.brain import Parameter
    from aria.brain.transformer import RMSNorm
    from aria.evaluation import check_gradients

    inputs = Parameter([[0.3, -0.7], [1.1, 0.2]])
    norm = RMSNorm(2)
    # Non-unit scale exposes errors that can be hidden by the initial all-ones weight.
    norm.weight._values = [0.7, 1.3]
    norm.weight.data = [0.7, 1.3]

    def loss():
        output = norm.forward(inputs)
        return (output * output).sum()

    report = check_gradients(loss, [inputs, norm.weight], max_checks=None)

    assert report.passed
    assert report.checked_values == 6
    assert report.max_absolute_error < 1e-5


def test_causal_attention_gradients_match_finite_differences() -> None:
    from aria.brain import Parameter
    from aria.brain.transformer import CausalSelfAttention
    from aria.evaluation import check_gradients

    inputs = Parameter([[0.2, -0.4], [0.7, 0.1]])
    attention = CausalSelfAttention(hidden_size=2, seed=31)

    def loss():
        output = attention.forward(inputs)
        return (output * output).sum()

    parameters = [inputs, *attention.parameters()]
    report = check_gradients(loss, parameters, max_checks=None)

    assert report.passed
    assert report.checked_values == sum(len(parameter._values) for parameter in parameters)
    assert report.max_absolute_error < 1e-5

def test_transformer_prefix_logits_are_independent_of_future_tokens() -> None:
    model = TransformerLanguageModel(
        vocab_size=7,
        hidden_size=4,
        intermediate_size=8,
        num_layers=2,
        max_sequence_length=8,
        seed=19,
    )

    prefix_logits = model.forward([1, 2])._values
    extended_logits = model.forward([1, 2, 3])._values

    assert prefix_logits == extended_logits[: 2 * model.vocab_size]

