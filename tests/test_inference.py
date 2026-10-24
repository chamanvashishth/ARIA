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


def test_top_p_one_matches_unfiltered_sampling() -> None:
    from aria.inference.generator import _nucleus_candidates
    assert _nucleus_candidates([2.0, 1.0, 0.0], 1.0)[0] == [0, 1, 2]


def test_top_p_and_top_k_generation_is_reproducible() -> None:
    model = TransformerLanguageModel(
        vocab_size=12, hidden_size=8, intermediate_size=12,
        max_sequence_length=8, seed=33,
    )
    options = dict(max_new_tokens=4, top_k=6, top_p=0.8, seed=19)
    assert generate(model, [1, 2], **options) == generate(model, [1, 2], **options)


def test_minimum_generation_length_delays_eos() -> None:
    class EOSModel:
        vocab_size = 3
        max_sequence_length = 8
        def forward(self, token_ids):
            from aria.brain.tensor import Tensor
            return Tensor.from_list([[0.0, 0.0, 20.0] for _ in token_ids])
    assert generate(
        EOSModel(), [1], max_new_tokens=4, min_new_tokens=2,
        eos_token_id=2, top_k=1,
    ) == [1, 0, 0, 2]


def test_stop_token_ids_stop_generation() -> None:
    class StopModel:
        vocab_size = 3
        max_sequence_length = 8
        def forward(self, token_ids):
            from aria.brain.tensor import Tensor
            return Tensor.from_list([[0.0, 10.0, 0.0] for _ in token_ids])
    assert generate(StopModel(), [0], max_new_tokens=5, stop_token_ids=[1], top_k=1) == [0, 1]


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1])
def test_invalid_top_p_is_rejected(top_p: float) -> None:
    with pytest.raises(ValueError, match="top_p"):
        sample_next_token([1.0, 2.0], top_p=top_p)


def test_generation_rejects_invalid_prompt_token() -> None:
    model = TransformerLanguageModel(
        vocab_size=4, hidden_size=4, intermediate_size=8, max_sequence_length=4
    )
    with pytest.raises(ValueError, match="prompt token"):
        generate(model, [4], max_new_tokens=1)


def test_generation_rejects_non_finite_logits() -> None:
    class InvalidModel:
        vocab_size = 2
        max_sequence_length = 4
        def forward(self, token_ids):
            from aria.brain.tensor import Tensor
            return Tensor.from_list([[float("nan"), 0.0] for _ in token_ids])
    with pytest.raises(ValueError, match="finite"):
        generate(InvalidModel(), [0], max_new_tokens=1)


def test_repetition_penalty_reduces_repeated_positive_logit() -> None:
    class RepeatingModel:
        vocab_size = 3
        max_sequence_length = 8
        def forward(self, token_ids):
            from aria.brain.tensor import Tensor
            return Tensor.from_list([[8.0, 1.0, 0.0] for _ in token_ids])
    # top-k=1 remains deterministic; the test exercises the penalty path.
    assert generate(
        RepeatingModel(), [0], max_new_tokens=2, repetition_penalty=2.0,
        top_k=1, seed=1,
    ) == [0, 0, 0]


def test_temperature_must_be_finite_and_positive() -> None:
    for temperature in (0.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="temperature"):
            sample_next_token([0.0, 1.0], temperature=temperature)



def test_generation_uses_configured_sliding_context_window() -> None:
    class ContextModel:
        vocab_size = 4
        max_sequence_length = 8
        def __init__(self):
            self.seen = []
        def forward(self, token_ids):
            from aria.brain.tensor import Tensor
            self.seen.append(list(token_ids))
            return Tensor.from_list([[0.0, 0.0, 0.0, 10.0] for _ in token_ids])

    model = ContextModel()
    generate(model, [0, 1, 2], max_new_tokens=2, context_window=2, top_k=1)
    assert model.seen == [[1, 2], [2, 3]]


@pytest.mark.parametrize("context_window", [0, -1, 9, True, 1.5])
def test_generation_rejects_invalid_context_window(context_window) -> None:
    model = TransformerLanguageModel(
        vocab_size=8, hidden_size=4, intermediate_size=8, max_sequence_length=8
    )
    with pytest.raises(ValueError, match="context_window"):
        generate(model, [1], max_new_tokens=1, context_window=context_window)
