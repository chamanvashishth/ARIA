"""Reproducible, dependency-free local autoregressive inference benchmark.

Example:
    python scripts/benchmark_inference.py --warmup 2 --iterations 10 --new-tokens 8

Results are measurements from the current machine and Python runtime, not
portable performance claims. This script benchmarks an untrained model.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

from aria.brain import TransformerLanguageModel


def compare_batch_execution(
    model: TransformerLanguageModel,
    sequences: list[list[int]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Compare batched forward execution with the same inputs run individually."""
    if not sequences or not sequences[0]:
        raise ValueError("sequences must be a non-empty batch of non-empty sequences")
    sequence_length = len(sequences[0])
    if any(len(sequence) != sequence_length for sequence in sequences):
        raise ValueError("all sequences must have the same length")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    def individual_forward() -> None:
        for sequence in sequences:
            model.forward(sequence)

    def batched_forward() -> None:
        model.forward_batch(sequences)

    individual_logits = [model.forward(sequence)._values for sequence in sequences]
    batched = model.forward_batch(sequences)
    vocab_size = model.vocab_size
    max_absolute_error = max(
        (abs(individual_logits[index][offset] - batched._values[index * sequence_length * vocab_size + offset])
         for index in range(len(sequences))
         for offset in range(sequence_length * vocab_size)),
        default=0.0,
    )

    for _ in range(warmup):
        individual_forward()
        batched_forward()

    def measure(run) -> list[float]:
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            run()
            samples.append(time.perf_counter() - started)
        return samples

    individual_times = measure(individual_forward)
    batched_times = measure(batched_forward)
    individual_median = statistics.median(individual_times)
    batched_median = statistics.median(batched_times)
    return {
        "batch_size": len(sequences),
        "sequence_length": sequence_length,
        "iterations": iterations,
        "warmup_iterations": warmup,
        "correctness": {"max_absolute_logit_error": max_absolute_error, "matches_within_1e-9": max_absolute_error <= 1e-9},
        "individual_execution": {
            "median_milliseconds_per_batch": individual_median * 1000,
            "sequences_per_second": len(sequences) / statistics.mean(individual_times),
        },
        "batched_execution": {
            "median_milliseconds_per_batch": batched_median * 1000,
            "sequences_per_second": len(sequences) / statistics.mean(batched_times),
        },
        "median_speedup_ratio": individual_median / batched_median if batched_median > 0 else None,
        "notes": [
            "Forward-only comparison; excludes backward pass and optimizer updates.",
            "Speedup ratio above 1 means batched execution was faster in this run.",
            "Small pure-Python workloads can be noisy; repeat on the target machine.",
        ],
    }



def benchmark_batch_sweep(
    *,
    batch_sizes: list[int],
    sequence_lengths: list[int],
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    """Measure a batch-size/sequence-length grid with consistent model settings."""
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if not sequence_lengths or any(length <= 0 for length in sequence_lengths):
        raise ValueError("sequence_lengths must contain positive integers")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    results = []
    for sequence_length in sequence_lengths:
        model = TransformerLanguageModel(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            max_sequence_length=max(32, sequence_length),
            seed=seed,
        )
        for batch_size in batch_sizes:
            sequences = [
                [(row * 7 + col) % model.vocab_size for col in range(sequence_length)]
                for row in range(batch_size)
            ]
            result = compare_batch_execution(
                model, sequences, warmup=warmup, iterations=iterations
            )
            results.append(result)

    return {
        "benchmark": "aria_transformer_batch_sweep",
        "model": {
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_layers": 1,
            "trained": False,
            "seed": seed,
        },
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "results": results,
        "notes": [
            "Forward-only; excludes backward pass and optimizer updates.",
            "Compare ratios within this run; avoid treating small differences as universal.",
            "Logits are checked against individual execution for every grid point.",
        ],
    }


def parameter_count(model: TransformerLanguageModel) -> int:
    """Count scalar values in all trainable parameters."""
    return sum(len(parameter._values) for parameter in model.parameters())


def benchmark(
    *,
    warmup: int,
    iterations: int,
    prompt_length: int,
    new_tokens: int,
    seed: int,
    batch_size: int = 4,
    batch_sequence_length: int = 8,
) -> dict[str, object]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if prompt_length <= 0 or new_tokens <= 0:
        raise ValueError("prompt_length and new_tokens must be positive")
    if batch_size <= 0 or batch_sequence_length <= 0:
        raise ValueError("batch_size and batch_sequence_length must be positive")

    model = TransformerLanguageModel(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        max_sequence_length=max(32, prompt_length + new_tokens),
        seed=seed,
    )
    prompt = [index % model.vocab_size for index in range(prompt_length)]

    def run_once() -> None:
        generated = list(prompt)
        for step in range(new_tokens):
            context = generated[-model.max_sequence_length:]
            logits = model.forward(context)
            # Greedy decoding isolates model-forward latency from RNG variance.
            next_token = max(range(model.vocab_size), key=logits._values[-model.vocab_size:].__getitem__)
            generated.append(next_token)

    for _ in range(warmup):
        run_once()

    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        run_once()
        durations.append(time.perf_counter() - started)

    total_seconds = sum(durations)
    comparison_model = TransformerLanguageModel(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        max_sequence_length=max(32, batch_sequence_length),
        seed=seed,
    )
    sequences = [[(row * 7 + col) % comparison_model.vocab_size for col in range(batch_sequence_length)]
                 for row in range(batch_size)]
    batch_comparison = compare_batch_execution(
        comparison_model, sequences, warmup=warmup, iterations=iterations
    )
    return {
        "benchmark": "aria_local_autoregressive_inference",
        "model": {
            "class": type(model).__name__,
            "vocab_size": model.vocab_size,
            "hidden_size": model.hidden_size,
            "intermediate_size": 32,
            "num_layers": len(model.layers),
            "max_sequence_length": model.max_sequence_length,
            "parameter_count": parameter_count(model),
            "trained": False,
        },
        "workload": {
            "prompt_length": prompt_length,
            "new_tokens_per_iteration": new_tokens,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "seed": seed,
        },
        "measurements": {
            "total_seconds": total_seconds,
            "mean_seconds_per_iteration": statistics.mean(durations),
            "median_seconds_per_iteration": statistics.median(durations),
            "mean_milliseconds_per_generated_token": (
                total_seconds * 1000 / (iterations * new_tokens)
            ),
            "tokens_per_second": (iterations * new_tokens / total_seconds)
            if total_seconds > 0 else None,
        },
        "batch_execution_comparison": batch_comparison,
        "environment": {
            "python_version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "notes": [
            "Untrained model; this measures execution speed, not output quality.",
            "Results depend on hardware, OS, Python version, and workload.",
            "No external inference API or third-party benchmark dependency is used.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-sequence-length", type=int, default=8)
    parser.add_argument("--sweep", action="store_true", help="also benchmark batch sizes 1,2,4,8 across sequence lengths 4,8,16")
    args = parser.parse_args()
    report = benchmark(
        warmup=args.warmup,
        iterations=args.iterations,
        prompt_length=args.prompt_length,
        new_tokens=args.new_tokens,
        seed=args.seed,
        batch_size=args.batch_size,
        batch_sequence_length=args.batch_sequence_length,
    )
    if args.sweep:
        report["batch_sweep"] = benchmark_batch_sweep(
            batch_sizes=[1, 2, 4, 8],
            sequence_lengths=[4, 8, 16],
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
