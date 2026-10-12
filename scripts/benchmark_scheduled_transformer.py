"""Compare direct and scheduler-integrated Transformer forward execution.

All backends run on the host CPU. Results are diagnostic and are not claims of
hardware acceleration or universal performance improvement.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

from aria.brain import TransformerLanguageModel
from aria.compute import ComputeRuntime, ScheduledComputeBackend


def _model(backend: object) -> TransformerLanguageModel:
    return TransformerLanguageModel(
        vocab_size=260,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        max_sequence_length=32,
        seed=123,
        backend=backend,  # type: ignore[arg-type]
    )


def _measure(model: TransformerLanguageModel, tokens: list[int], iterations: int) -> tuple[float, list[float]]:
    samples: list[float] = []
    output: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        logits = model.forward(tokens)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
        output = logits._values
    return statistics.median(samples), output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.sequence_length <= 0:
        parser.error("iterations and sequence-length must be positive; warmup cannot be negative")

    tokens = [index % 260 for index in range(args.sequence_length)]
    direct_model = _model(ComputeRuntime("vcpu"))
    scheduled_backend = ScheduledComputeBackend(ComputeRuntime("vcpu"))
    scheduled_model = _model(scheduled_backend)

    for _ in range(args.warmup):
        direct_model.forward(tokens)
        scheduled_model.forward(tokens)
    direct_ms, direct_output = _measure(direct_model, tokens, args.iterations)
    scheduled_ms, scheduled_output = _measure(scheduled_model, tokens, args.iterations)

    report = {
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "workload": {
            "sequence_length": args.sequence_length,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "model": "untrained one-layer Transformer, hidden=16, intermediate=32, vocab=260",
        },
        "execution": {
            "direct_vcpu_median_ms": direct_ms,
            "scheduled_vcpu_median_ms": scheduled_ms,
            "scheduled_matmul_commands": scheduled_backend.stats.submitted,
            "scheduled_completed_commands": scheduled_backend.stats.completed,
            "scheduled_peak_estimated_memory_bytes": scheduled_backend.stats.peak_reserved_memory_bytes,
            "numerically_equivalent": len(direct_output) == len(scheduled_output)
            and all(abs(a - b) <= 1e-12 for a, b in zip(direct_output, scheduled_output)),
            "hardware_acceleration": False,
        },
        "limitations": [
            "All computation executes on the host CPU.",
            "The scheduler adapter submits and completes each matrix command synchronously.",
            "Timings are diagnostic and are not a performance guarantee.",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
