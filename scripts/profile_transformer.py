"""Profile ARIA Transformer forward and backward costs on the current machine.

Example:
    python scripts/profile_transformer.py --iterations 5 --warmup 1 --batch-size 2 --sequence-length 8

Timings are diagnostic measurements, not portable performance claims.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path
import time
from collections.abc import Callable

from aria.brain import TransformerLanguageModel
from aria.brain.layers import Linear
from aria.brain.tensor import Tensor
from aria.brain.transformer import CausalSelfAttention


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples) * 1000,
        "mean_ms": statistics.mean(samples) * 1000,
    }


def _measure(operation: Callable[[], None], *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return samples


def _profile_backward(forward_loss: Callable[[], Tensor], zero_grad: Callable[[], None],
                      *, warmup: int, iterations: int) -> dict[str, dict[str, float]]:
    forward_samples: list[float] = []
    backward_samples: list[float] = []
    for index in range(warmup + iterations):
        zero_grad()
        started = time.perf_counter()
        loss = forward_loss()
        forward_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        loss.backward()
        backward_elapsed = time.perf_counter() - started
        if index >= warmup:
            forward_samples.append(forward_elapsed)
            backward_samples.append(backward_elapsed)
    return {"forward_and_loss": _summary(forward_samples), "backward": _summary(backward_samples)}


def profile_transformer_components(
    *, batch_size: int = 2, sequence_length: int = 8, iterations: int = 5,
    warmup: int = 1, seed: int = 29,
) -> dict[str, object]:
    """Time Linear, causal attention, and a full Transformer training step."""
    if batch_size <= 0 or sequence_length <= 0 or iterations <= 0 or warmup < 0:
        raise ValueError("batch_size, sequence_length, and iterations must be positive; warmup must be non-negative")

    hidden, intermediate, vocab = 16, 32, 64
    linear = Linear(hidden, intermediate, seed=seed)
    linear_input = Tensor(
        [[[((b * sequence_length + t + d) % 13) / 13 for d in range(hidden)]
          for t in range(sequence_length)] for b in range(batch_size)],
        requires_grad=True,
    )
    linear_profile = _profile_backward(
        lambda: linear.forward(linear_input).sum(),
        lambda: (linear.zero_grad(), linear_input.zero_grad()),
        warmup=warmup, iterations=iterations,
    )

    attention = CausalSelfAttention(hidden, seed=seed)
    attention_input = Tensor(
        [[[((b * sequence_length + t + d) % 11) / 11 for d in range(hidden)]
          for t in range(sequence_length)] for b in range(batch_size)],
        requires_grad=True,
    )
    attention_profile = _profile_backward(
        lambda: attention.forward(attention_input).sum(),
        lambda: (attention.zero_grad(), attention_input.zero_grad()),
        warmup=warmup, iterations=iterations,
    )

    model = TransformerLanguageModel(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=intermediate,
        num_layers=1, max_sequence_length=max(32, sequence_length), seed=seed,
    )
    inputs = [[(row * 7 + col) % vocab for col in range(sequence_length)]
              for row in range(batch_size)]
    targets = [[(token + 1) % vocab for token in row] for row in inputs]
    model_profile = _profile_backward(
        lambda: model.loss_batch(inputs, targets),
        model.zero_grad,
        warmup=warmup, iterations=iterations,
    )
    return {
        "benchmark": "aria_transformer_component_profile",
        "configuration": {
            "batch_size": batch_size, "sequence_length": sequence_length,
            "hidden_size": hidden, "intermediate_size": intermediate,
            "vocab_size": vocab, "iterations": iterations, "warmup": warmup, "seed": seed,
        },
        "components": {
            "linear_forward_and_backward": linear_profile,
            "causal_attention_forward_and_backward": attention_profile,
            "full_transformer_training_step": model_profile,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "notes": [
            "Forward timing includes scalar reduction for isolated Linear and attention components.",
            "Full-model forward timing includes batched logits and mean next-token loss.",
            "Backward timing starts after the forward graph and loss have been built.",
            "No optimizer update is included; measurements are local and workload-dependent.",
        ],
    }



def profile_transformer_sweep(
    *,
    batch_sizes: list[int],
    sequence_lengths: list[int],
    iterations: int = 5,
    warmup: int = 1,
    seed: int = 29,
) -> dict[str, object]:
    """Profile a grid of batch sizes and sequence lengths."""
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if not sequence_lengths or any(length <= 0 for length in sequence_lengths):
        raise ValueError("sequence_lengths must contain positive integers")
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup must be non-negative")

    results = []
    for sequence_length in sequence_lengths:
        for batch_size in batch_sizes:
            results.append(profile_transformer_components(
                batch_size=batch_size,
                sequence_length=sequence_length,
                iterations=iterations,
                warmup=warmup,
                seed=seed,
            ))
    return {
        "benchmark": "aria_transformer_component_profile_sweep",
        "batch_sizes": batch_sizes,
        "sequence_lengths": sequence_lengths,
        "iterations": iterations,
        "warmup": warmup,
        "results": results,
        "notes": [
            "Each configuration creates a fresh deterministically seeded model.",
            "Compare component timings across configurations on the same machine.",
            "This sweep profiles diagnostic small models, not production workloads.",
        ],
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--sweep", action="store_true",
        help="also profile batch sizes 1,2,4 across sequence lengths 4,8,16",
    )
    parser.add_argument(
        "--output", type=Path,
        help="write the JSON report to this file instead of standard output",
    )
    args = parser.parse_args()
    report = profile_transformer_components(
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        iterations=args.iterations, warmup=args.warmup, seed=args.seed,
    )
    if args.sweep:
        report["sweep"] = profile_transformer_sweep(
            batch_sizes=[1, 2, 4],
            sequence_lengths=[4, 8, 16],
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
        )
    rendered = json.dumps(report, indent=2) + "\\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
