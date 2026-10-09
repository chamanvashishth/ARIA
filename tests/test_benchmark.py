import pytest

from aria.brain import TransformerLanguageModel
from scripts.benchmark_inference import benchmark_batch_sweep, compare_batch_execution


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



def test_batch_sweep_covers_grid_and_checks_logits() -> None:
    report = benchmark_batch_sweep(
        batch_sizes=[1, 2],
        sequence_lengths=[2, 3],
        warmup=0,
        iterations=1,
        seed=73,
    )

    assert len(report["results"]) == 4
    assert {
        (item["batch_size"], item["sequence_length"])
        for item in report["results"]
    } == {(1, 2), (2, 2), (1, 3), (2, 3)}
    assert all(item["correctness"]["matches_within_1e-9"] for item in report["results"])


def test_batch_sweep_rejects_invalid_grid() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        benchmark_batch_sweep(
            batch_sizes=[1, 0],
            sequence_lengths=[2],
            warmup=0,
            iterations=1,
            seed=74,
        )



def test_transformer_component_profiler_reports_forward_and_backward() -> None:
    from scripts.profile_transformer import profile_transformer_components

    report = profile_transformer_components(
        batch_size=1,
        sequence_length=2,
        iterations=1,
        warmup=0,
        seed=75,
    )

    assert report["benchmark"] == "aria_transformer_component_profile"
    assert set(report["components"]) == {
        "linear_forward_and_backward",
        "causal_attention_forward_and_backward",
        "full_transformer_training_step",
    }
    for component in report["components"].values():
        assert component["forward_and_loss"]["median_ms"] >= 0
        assert component["backward"]["median_ms"] >= 0


def test_transformer_component_profiler_rejects_invalid_dimensions() -> None:
    from scripts.profile_transformer import profile_transformer_components

    with pytest.raises(ValueError, match="must be positive"):
        profile_transformer_components(batch_size=0, sequence_length=2, iterations=1, warmup=0)



def test_transformer_profiler_sweep_covers_grid() -> None:
    from scripts.profile_transformer import profile_transformer_sweep

    report = profile_transformer_sweep(
        batch_sizes=[1, 2],
        sequence_lengths=[2, 3],
        iterations=1,
        warmup=0,
        seed=77,
    )

    assert len(report["results"]) == 4
    assert {
        (item["configuration"]["batch_size"], item["configuration"]["sequence_length"])
        for item in report["results"]
    } == {(1, 2), (2, 2), (1, 3), (2, 3)}
    assert all(
        item["components"]["full_transformer_training_step"]["backward"]["median_ms"] >= 0
        for item in report["results"]
    )


def test_transformer_profiler_sweep_rejects_invalid_grid() -> None:
    from scripts.profile_transformer import profile_transformer_sweep

    with pytest.raises(ValueError, match="positive integers"):
        profile_transformer_sweep(
            batch_sizes=[1],
            sequence_lengths=[0],
            iterations=1,
            warmup=0,
        )
