import pytest

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



def test_transformer_batched_logits_match_independent_forward_passes() -> None:
    model = TransformerLanguageModel(
        vocab_size=9,
        hidden_size=4,
        intermediate_size=8,
        num_layers=2,
        max_sequence_length=5,
        seed=41,
    )
    sequences = [[1, 2, 3], [4, 5, 6]]
    batched = model.forward_batch(sequences)

    assert batched.shape == (2, 3, 9)
    for index, sequence in enumerate(sequences):
        individual = model.forward(sequence)
        start = index * 3 * model.vocab_size
        assert batched._values[start:start + 3 * model.vocab_size] == individual._values


def test_transformer_batched_attention_has_no_cross_sample_leakage() -> None:
    model = TransformerLanguageModel(
        vocab_size=9,
        hidden_size=4,
        intermediate_size=8,
        num_layers=2,
        max_sequence_length=5,
        seed=42,
    )
    first = [1, 2, 3]
    batch_a = model.forward_batch([first, [4, 5, 6]])
    batch_b = model.forward_batch([first, [7, 8, 1]])

    width = len(first) * model.vocab_size
    assert batch_a._values[:width] == batch_b._values[:width]


def test_transformer_batched_loss_and_gradients_match_individual_mean() -> None:
    from aria.brain import SGD

    sequences = [[1, 2, 3], [4, 5, 6]]
    targets = [[2, 3, 4], [5, 6, 7]]
    batched_model = TransformerLanguageModel(
        vocab_size=9,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        max_sequence_length=5,
        seed=43,
    )
    individual_model = TransformerLanguageModel(
        vocab_size=9,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        max_sequence_length=5,
        seed=43,
    )

    batch_loss = batched_model.loss_batch(sequences, targets)
    batch_loss.backward()
    individual_losses = [
        individual_model.loss_batch([sequence], [target])
        for sequence, target in zip(sequences, targets)
    ]
    for loss in individual_losses:
        loss.backward()
    for parameter in individual_model.parameters():
        assert parameter.grad is not None
        parameter.grad._values = [value / len(individual_losses) for value in parameter.grad._values]
        parameter.grad.data = parameter.grad.to_list()

    assert batch_loss.item() == pytest.approx(
        sum(loss.item() for loss in individual_losses) / len(individual_losses)
    )
    for batched_parameter, individual_parameter in zip(
        batched_model.parameters(), individual_model.parameters()
    ):
        assert batched_parameter.grad is not None
        assert individual_parameter.grad is not None
        assert batched_parameter.grad._values == __import__("pytest").approx(
            individual_parameter.grad._values, abs=1e-8, rel=1e-7
        )


def test_transformer_batched_gradient_checker_passes() -> None:
    from aria.evaluation import check_gradients

    model = TransformerLanguageModel(
        vocab_size=4,
        hidden_size=2,
        intermediate_size=3,
        num_layers=1,
        max_sequence_length=3,
        seed=44,
    )
    report = check_gradients(
        lambda: model.loss_batch([[1, 2], [2, 3]], [[2, 3], [3, 1]]),
        model.parameters(),
        max_checks=8,
    )

    assert report.passed
    assert report.checked_values == 8



def test_batched_causal_attention_gradients_match_finite_differences() -> None:
    from aria.brain import Parameter
    from aria.brain.transformer import CausalSelfAttention
    from aria.evaluation import check_gradients

    inputs = Parameter([
        [[0.2, -0.4], [0.7, 0.1]],
        [[-0.3, 0.5], [0.9, -0.2]],
    ])
    attention = CausalSelfAttention(hidden_size=2, seed=76)

    def loss():
        output = attention.forward(inputs)
        return (output * output).sum()

    parameters = [inputs, *attention.parameters()]
    report = check_gradients(loss, parameters, max_checks=None)

    assert report.passed
    assert report.checked_values == sum(len(parameter._values) for parameter in parameters)
    assert report.max_absolute_error < 1e-5



def test_feed_forward_parameter_gradients_match_finite_differences() -> None:
    from aria.brain import Parameter
    from aria.evaluation import check_gradients
    from aria.brain.transformer import FeedForward

    inputs = Parameter([[0.2, -0.5], [0.8, 0.3]])
    feed_forward = FeedForward(hidden_size=2, intermediate_size=3, seed=91)

    # Chosen inputs and deterministic weights keep the hidden activations away
    # from the ReLU kink so central differences test the smooth path.
    def loss():
        output = feed_forward.forward(inputs)
        return (output * output).sum()

    parameters = [inputs, *feed_forward.parameters()]
    report = check_gradients(
        loss,
        parameters,
        epsilon=1e-5,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
        max_checks=None,
    )

    assert report.passed
    assert report.checked_values == sum(len(parameter._values) for parameter in parameters)
    assert report.max_absolute_error < 1e-5


def test_transformer_block_gradients_match_finite_differences() -> None:
    from aria.brain import Parameter
    from aria.evaluation import check_gradients
    from aria.brain.transformer import TransformerBlock

    inputs = Parameter([[0.2, -0.4], [0.7, 0.1]])
    block = TransformerBlock(hidden_size=2, intermediate_size=3, seed=92)

    def loss():
        output = block.forward(inputs)
        return (output * output).sum()

    parameters = [inputs, *block.parameters()]
    report = check_gradients(
        loss,
        parameters,
        epsilon=1e-5,
        absolute_tolerance=2e-5,
        relative_tolerance=2e-4,
        max_checks=None,
    )

    assert report.passed
    assert report.checked_values == sum(len(parameter._values) for parameter in parameters)
    assert report.max_absolute_error < 2e-5



def test_transformer_language_model_all_parameter_gradients_match_finite_differences() -> None:
    from aria.evaluation import check_gradients

    # Keep the model tiny so every scalar parameter can be checked exhaustively.
    model = TransformerLanguageModel(
        vocab_size=3,
        hidden_size=2,
        intermediate_size=2,
        num_layers=1,
        max_sequence_length=2,
        seed=123,
    )

    report = check_gradients(
        lambda: model.loss_batch([[0, 1]], [[1, 2]]),
        model.parameters(),
        epsilon=1e-5,
        absolute_tolerance=3e-5,
        relative_tolerance=3e-4,
        max_checks=None,
    )

    assert report.passed
    assert report.checked_values == sum(
        len(parameter._values) for parameter in model.parameters()
    )
    assert report.max_absolute_error < 3e-5
