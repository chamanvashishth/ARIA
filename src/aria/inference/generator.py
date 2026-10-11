"""Local autoregressive text generation for ARIA."""

from __future__ import annotations

import math
import random
from typing import Protocol


class LanguageModel(Protocol):
    vocab_size: int
    max_sequence_length: int

    def forward(self, token_ids: list[int]): ...


def _validate_logits(logits: list[float]) -> None:
    if not logits:
        raise ValueError("logits cannot be empty")
    if any(not math.isfinite(value) for value in logits):
        raise ValueError("logits must contain only finite values")


def _softmax(logits: list[float], temperature: float) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _nucleus_candidates(
    logits: list[float],
    top_p: float,
    temperature: float,
) -> tuple[list[int], list[float]]:
    """Return the smallest highest-probability prefix reaching top_p."""
    probabilities = _softmax(logits, temperature)
    ordered = sorted(range(len(logits)), key=lambda index: probabilities[index], reverse=True)
    selected: list[int] = []
    cumulative = 0.0
    for index in ordered:
        selected.append(index)
        cumulative += probabilities[index]
        if cumulative >= top_p:
            break
    selected_total = sum(probabilities[index] for index in selected)
    return selected, [probabilities[index] / selected_total for index in selected]


def _sample(probs: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probs):
        cumulative += probability
        if threshold < cumulative:
            return index
    return len(probs) - 1


def sample_next_token(
    logits: list[float],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    rng: random.Random | None = None,
) -> int:
    """Sample from logits with temperature, optional top-k, and nucleus top-p."""
    _validate_logits(logits)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(logits)
    ):
        raise ValueError("top_k must be between 1 and the vocabulary size")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    rng = rng or random.Random()

    candidates = list(range(len(logits)))
    candidate_logits = list(logits)
    if top_k is not None and top_k < len(logits):
        candidates = sorted(candidates, key=logits.__getitem__, reverse=True)[:top_k]
        candidate_logits = [logits[index] for index in candidates]

    if top_p < 1.0:
        nucleus, probabilities = _nucleus_candidates(candidate_logits, top_p, temperature)
        return candidates[nucleus[_sample(probabilities, rng)]]

    probabilities = _softmax(candidate_logits, temperature)
    return candidates[_sample(probabilities, rng)]


def _validate_token_id(token_id: int, vocab_size: int, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} must be an integer token ID")
    if not 0 <= token_id < vocab_size:
        raise ValueError(f"{label} is outside the model vocabulary")


def generate(
    model: LanguageModel,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
    min_new_tokens: int = 0,
    eos_token_id: int | None = None,
    stop_token_ids: list[int] | None = None,
    seed: int | None = None,
) -> list[int]:
    """Generate tokens autoregressively with configurable sampling controls.

    top_k is applied before top_p. Repetition penalty is applied to tokens
    present in the recent context window (or the full generated context).
    """
    if not prompt_tokens:
        raise ValueError("prompt_tokens cannot be empty")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if isinstance(min_new_tokens, bool) or not isinstance(min_new_tokens, int):
        raise ValueError("min_new_tokens must be a non-negative integer")
    if not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("min_new_tokens must be between zero and max_new_tokens")
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be finite and at least 1.0")
    if repetition_window is not None and (
        isinstance(repetition_window, bool)
        or not isinstance(repetition_window, int)
        or repetition_window <= 0
    ):
        raise ValueError("repetition_window must be a positive integer")
    if len(prompt_tokens) > model.max_sequence_length:
        raise ValueError("prompt exceeds model context length")
    for token in prompt_tokens:
        _validate_token_id(token, model.vocab_size, "prompt token")
    if eos_token_id is not None:
        _validate_token_id(eos_token_id, model.vocab_size, "eos_token_id")
    stop_ids = set(stop_token_ids or [])
    if eos_token_id is not None:
        stop_ids.add(eos_token_id)
    for token_id in stop_ids:
        _validate_token_id(token_id, model.vocab_size, "stop token")

    generated = list(prompt_tokens)
    rng = random.Random(seed)

    for step in range(max_new_tokens):
        context = generated[-model.max_sequence_length:]
        logits_tensor = model.forward(context)
        logits = list(logits_tensor.to_list()[-1])
        if len(logits) != model.vocab_size:
            raise ValueError("model output size does not match vocabulary size")
        _validate_logits(logits)

        if repetition_penalty > 1.0:
            recent = generated if repetition_window is None else generated[-repetition_window:]
            for token_id in set(recent):
                logits[token_id] = (
                    logits[token_id] / repetition_penalty
                    if logits[token_id] > 0
                    else logits[token_id] * repetition_penalty
                )

        if eos_token_id is not None and step < min_new_tokens:
            # Keep EOS out of the candidate set until the minimum length is met.
            logits[eos_token_id] = min(logits) - max(1.0, abs(min(logits)))

        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
        )
        generated.append(next_token)
        if next_token in stop_ids:
            break

    return generated
