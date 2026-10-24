"""Compare ARIA's software compute backends on a deterministic matrix workload.

Example:
    python scripts/benchmark_compute.py --size 32 --warmup 1 --iterations 5

All devices are host-CPU emulators. Results are local diagnostics, not
portable performance claims or evidence of physical GPU/NPU acceleration.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable

from aria.compute import ComputeRuntime, DeviceKind


def _matrix(rows: int, columns: int, *, offset: int) -> list[list[float]]:
    """Create stable, bounded values without random-number dependencies."""
    return [
        [(((row * 17 + column * 11 + offset) % 29) - 14) / 14 for column in range(columns)]
        for row in range(rows)
    ]


def _error_metrics(
    actual: list[list[float]], expected: list[list[float]]
) -> dict[str, float]:
    differences = [
        actual[row][column] - expected[row][column]
        for row in range(len(expected))
        for column in range(len(expected[0]))
    ]
    squared_error = sum(value * value for value in differences)
    squared_reference = sum(
        value * value for row in expected for value in row
    )
    return {
        "max_absolute_error": max((abs(value) for value in differences), default=0.0),
        "relative_l2_error": (
            math.sqrt(squared_error / squared_reference)
            if squared_reference > 0 else math.sqrt(squared_error)
        ),
    }


def _measure(
    operation: Callable[[], list[list[float]]],
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[list[float]], list[float]]:
    for _ in range(warmup):
        operation()
    samples: list[float] = []
    output: list[list[float]] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        output = operation()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return output, samples


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def benchmark_compute(
    *,
    rows: int = 32,
    inner: int = 32,
    columns: int = 32,
    warmup: int = 1,
    iterations: int = 5,
    tile_size: int = 16,
) -> dict[str, object]:
    """Measure all devices on identical inputs and compare with VCPU."""
    if min(rows, inner, columns, iterations, tile_size) <= 0 or warmup < 0:
        raise ValueError("matrix dimensions, iterations and tile_size must be positive; warmup cannot be negative")

    left = _matrix(rows, inner, offset=3)
    right = _matrix(inner, columns, offset=19)
    devices: tuple[DeviceKind, ...] = ("vcpu", "vgpu", "vnpu")
    results: dict[str, object] = {}
    reference: list[list[float]] | None = None

    for device in devices:
        runtime = ComputeRuntime(device, tile_size=tile_size)
        output, samples = _measure(
            lambda runtime=runtime: runtime.matmul(left, right),
            warmup=warmup,
            iterations=iterations,
        )
        if device == "vcpu":
            reference = output
            errors = {"max_absolute_error": 0.0, "relative_l2_error": 0.0}
        else:
            assert reference is not None
            errors = _error_metrics(output, reference)

        results[device] = {
            "backend": {
                "name": runtime.info.name,
                "execution": runtime.info.execution,
                "numeric_mode": runtime.info.numeric_mode,
                "hardware_acceleration": runtime.info.hardware_acceleration,
            },
            "latency_ms": {
                "median": statistics.median(samples),
                "mean": statistics.mean(samples),
                "min": min(samples),
                "p95": _percentile(samples, 0.95),
            },
            "numerical_error_vs_vcpu": errors,
            "iterations": iterations,
            "warmup_iterations": warmup,
        }

    return {
        "benchmark": "aria_compute_backend_comparison",
        "workload": {
            "left_shape": [rows, inner],
            "right_shape": [inner, columns],
            "multiply_accumulate_count": rows * inner * columns,
            "tile_size": tile_size,
        },
        "results": results,
        "environment": {
            "python_version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "interpretation": [
            "All backends execute on the host CPU; no physical accelerator is used.",
            "VCPU is the floating-point numerical reference.",
            "VNPU uses per-matrix symmetric INT8 quantization; non-zero numerical error is expected.",
            "Small workloads and pure-Python loops can be noisy; repeat on the target machine.",
            "Latency results describe this machine and workload only; no universal speedup is implied.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--inner", type=int, default=32)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=16)
    args = parser.parse_args()
    report = benchmark_compute(
        rows=args.rows,
        inner=args.inner,
        columns=args.columns,
        warmup=args.warmup,
        iterations=args.iterations,
        tile_size=args.tile_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
