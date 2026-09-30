import pytest

from aria.brain import TransformerLanguageModel
from aria.inference import generate, sample_next_token


def test_top_k_one_is_deterministic() -> None:
    assert sample_next_token([0.1, 3.0, 2.0], top_k=1) == 1


def test_generation_is_reproducible_with_seed() -> None:
    model = TransformerLanguageModel(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        max_sequence_length=8,
        seed=10,
    )
    first = generate(model, [1, 2], max_new_tokens=3, top_k=4, seed=7)
    second = generate(model, [1, 2], max_new_tokens=3, top_k=4, seed=7)
    assert first == second
    assert len(first) == 5


def test_generation_stops_at_eos() -> None:
    class EOSModel:
        vocab_size = 4
        max_sequence_length = 4

        def forward(self, token_ids):
            from aria.brain.tensor import Tensor
            return Tensor.from_list([[0.0, 0.0, 0.0, 10.0] for _ in token_ids])

    result = generate(EOSModel(), [1], max_new_tokens=10, top_k=1, eos_token_id=3)
    assert result == [1, 3]


def test_generation_rejects_long_prompt() -> None:
    model = TransformerLanguageModel(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        max_sequence_length=2,
    )
    with pytest.raises(ValueError, match="context"):
        generate(model, [1, 2, 3], max_new_tokens=1)
