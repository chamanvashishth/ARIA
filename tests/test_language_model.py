import math

import pytest

from aria.brain import SGD, TinyLanguageModel


def test_language_model_shapes() -> None:
    model = TinyLanguageModel(vocab_size=8, embedding_dim=4, seed=1)
    logits = model.forward([1, 2, 3])
    assert logits.shape == (3, 8)


def test_cross_entropy_is_positive_and_finite() -> None:
    model = TinyLanguageModel(vocab_size=6, embedding_dim=3, seed=2)
    loss = model.loss([1, 2], [2, 3])
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0


def test_language_model_backpropagates() -> None:
    model = TinyLanguageModel(vocab_size=5, embedding_dim=3, seed=3)
    loss = model.loss([1, 2, 1], [2, 1, 3])
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_optimizer_changes_parameters() -> None:
    model = TinyLanguageModel(vocab_size=5, embedding_dim=3, seed=4)
    optimizer = SGD(model.parameters(), learning_rate=0.1)
    before = list(model.vocabulary.weight._values)

    loss = model.loss([1, 2], [2, 3])
    loss.backward()
    optimizer.step()

    assert model.vocabulary.weight._values != before


def test_invalid_target_is_rejected() -> None:
    model = TinyLanguageModel(vocab_size=4, embedding_dim=2)
    with pytest.raises(IndexError):
        model.loss([1], [4])



def test_tiny_language_model_uses_injected_compute_backend() -> None:
    from aria.compute import VirtualGPUBackend

    model = TinyLanguageModel(vocab_size=5, embedding_dim=3, seed=301, backend=VirtualGPUBackend())
    logits = model.forward([1, 2, 3])
    assert logits.shape == (3, 5)

    loss = model.loss([1, 2, 3], [2, 3, 4])
    loss.backward()
    assert model.vocabulary.weight.grad is not None


def test_tiny_language_model_vnpu_supports_frozen_inference_only() -> None:
    from aria.compute import VirtualNPUBackend

    model = TinyLanguageModel(vocab_size=5, embedding_dim=3, seed=302, backend=VirtualNPUBackend())
    model.requires_grad_(False)
    logits = model.forward([1, 2, 3])

    assert logits.shape == (3, 5)
    assert logits.requires_grad is False
