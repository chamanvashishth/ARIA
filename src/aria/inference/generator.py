"""Local autoregressive text generation for ARIA."""

from __future__ import annotations

import math
import random
from typing import Protocol


class LanguageModel(Protocol):
    vocab_size: int
    max_sequence_length: int

    def forward(self, token_ids: list[int]): ...


def _softmax(logits: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _sample(probs: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probs):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probs) - 1


def sample_next_token(
    logits: list[float],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: random.Random | None = None,
) -> int:
    """Sample one token from logits using temperature and optional top-k."""
    if not logits:
        raise ValueError("logits cannot be empty")
    if top_k is not None and not 1 <= top_k <= len(logits):
        raise ValueError("top_k must be between 1 and the vocabulary size")
    rng = rng or random.Random()

    if top_k is not None and top_k < len(logits):
        candidates = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:top_k]
        candidate_logits = [logits[index] for index in candidates]
        candidate_probs = _softmax(candidate_logits, temperature)
        return candidates[_sample(candidate_probs, rng)]

    return _sample(_softmax(logits, temperature), rng)


def generate(
    model: LanguageModel,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    seed: int | None = None,
) -> list[int]:
    """Generate tokens autoregressively from a local language model."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not prompt_tokens:
        raise ValueError("prompt_tokens cannot be empty")
    if len(prompt_tokens) > model.max_sequence_length:
        raise ValueError("prompt exceeds model context length")

    generated = list(prompt_tokens)
    rng = random.Random(seed)

    for _ in range(max_new_tokens):
        context = generated[-model.max_sequence_length:]
        logits = model.forward(context)
        next_token = sample_next_token(
            logits.to_list()[-1],
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break

    return generated
