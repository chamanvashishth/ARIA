import pytest

from aria.brain import TransformerLanguageModel
from scripts.benchmark_inference import compare_batch_execution


def test_batch_benchmark_reports_equivalent_logits_and_positive_throughput() -> None:
    model = TransformerLanguageModel(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        max_sequence_length=4,
        seed=71,
    )

    report = compare_batch_execution(
        model,
        [[1, 2, 3], [4, 5, 6]],
        warmup=0,
        iterations=1,
    )

    assert report["correctness"]["matches_within_1e-9"] is True
    assert report["correctness"]["max_absolute_logit_error"] <= 1e-9
    assert report["individual_execution"]["sequences_per_second"] > 0
    assert report["batched_execution"]["sequences_per_second"] > 0
    assert report["median_speedup_ratio"] > 0


def test_batch_benchmark_rejects_mixed_sequence_lengths() -> None:
    model = TransformerLanguageModel(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        max_sequence_length=4,
        seed=72,
    )

    with pytest.raises(ValueError, match="same length"):
        compare_batch_execution(model, [[1, 2], [3]], warmup=0, iterations=1)
