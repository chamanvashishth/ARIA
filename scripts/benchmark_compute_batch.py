"""Compare individual scheduler calls with one shared-queue batch.

Example:
    python scripts/benchmark_compute_batch.py --batch-size 8 --size 16

Both paths use the same host-CPU VCPU runtime. This is a local diagnostic,
not a promise that batching is faster or that operations run in parallel.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

from aria.compute import ComputeRuntime, ScheduledComputeBackend


def _matrix(size: int, offset: int) -> list[list[float]]:
    return [
        [(((row * 13 + column * 7 + offset) % 23) - 11) / 11 for column in range(size)]
        for row in range(size)
    ]


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples)


def benchmark_batch(
    *, batch_size: int = 8, size: int = 16, warmup: int = 1, iterations: int = 5
) -> dict[str, object]:
    if min(batch_size, size, iterations) <= 0 or warmup < 0:
        raise ValueError("batch_size, size and iterations must be positive; warmup cannot be negative")

    operations = tuple(
        (_matrix(size, index), _matrix(size, index + 17))
        for index in range(batch_size)
    )
    individual = ScheduledComputeBackend(ComputeRuntime("vcpu"))
    batched = ScheduledComputeBackend(ComputeRuntime("vcpu"))

    def run_individual():
        return tuple(individual.matmul(left, right) for left, right in operations)

    def run_batch():
        return batched.matmul_batch(operations)

    for _ in range(warmup):
        run_individual()
        run_batch()

    individual_samples: list[float] = []
    batch_samples: list[float] = []
    reference = ()
    batch_output = ()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        reference = run_individual()
        individual_samples.append((time.perf_counter_ns() - started) / 1_000_000)

        started = time.perf_counter_ns()
        batch_output = run_batch()
        batch_samples.append((time.perf_counter_ns() - started) / 1_000_000)

    max_error = max(
        (
            abs(reference[op][row][column] - batch_output[op][row][column])
            for op in range(batch_size)
            for row in range(size)
            for column in range(size)
        ),
        default=0.0,
    )
    return {
        "benchmark": "aria_shared_queue_batch_comparison",
        "workload": {
            "batch_size": batch_size,
            "matrix_shape": [size, size],
            "operations_per_iteration": batch_size,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
        },
        "results": {
            "individual_calls": {
                "median_latency_ms": _median_ms(individual_samples),
                "commands_per_iteration": batch_size,
            },
            "shared_queue_batch": {
                "median_latency_ms": _median_ms(batch_samples),
                "commands_per_iteration": batch_size,
                "peak_reserved_memory_bytes": batched.stats.peak_reserved_memory_bytes,
            },
            "max_absolute_output_difference": max_error,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "interpretation": [
            "Both paths use the same VCPU host-CPU matrix runtime.",
            "The shared queue executes commands synchronously in FIFO order.",
            "Batching reduces queue construction to one queue per batch; it does not add parallel workers.",
            "Small synthetic workloads are noisy; no universal speedup is implied.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark_batch(
                batch_size=args.batch_size,
                size=args.size,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
