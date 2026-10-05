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
) -> dict[str, object]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if prompt_length <= 0 or new_tokens <= 0:
        raise ValueError("prompt_length and new_tokens must be positive")

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
    args = parser.parse_args()
    print(json.dumps(benchmark(
        warmup=args.warmup,
        iterations=args.iterations,
        prompt_length=args.prompt_length,
        new_tokens=args.new_tokens,
        seed=args.seed,
    ), indent=2))


if __name__ == "__main__":
    main()
