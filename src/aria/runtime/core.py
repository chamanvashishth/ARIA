"""Execution runtime boundary for ARIA's local neural stack."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from aria.inference import generate
from aria.tokenizer import ByteTokenizer


@dataclass(frozen=True)
class RuntimeConfig:
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float


class AriaRuntime:
    """Own model lifecycle and connect text I/O to local inference."""

    def __init__(self, model, tokenizer: ByteTokenizer | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer or ByteTokenizer()
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        """Start the local runtime without creating network dependencies."""
        self._started = True

    def stop(self) -> None:
        """Stop the runtime and release its active execution state."""
        self._started = False

    def generate(self, prompt: str, config: RuntimeConfig | None = None) -> GenerationResult:
        if not self._started:
            raise RuntimeError("ARIA runtime is not started")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")

        config = config or RuntimeConfig()
        prompt_tokens = self.tokenizer.encode(prompt)
        started = perf_counter()
        token_ids = generate(
            self.model,
            prompt_tokens,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            eos_token_id=self.tokenizer.EOS_ID,
            seed=config.seed,
        )
        elapsed = perf_counter() - started
        generated_tokens = token_ids[len(prompt_tokens):]
        return GenerationResult(
            text=self.tokenizer.decode(token_ids),
            token_ids=token_ids,
            prompt_tokens=len(prompt_tokens),
            generated_tokens=len(generated_tokens),
            elapsed_seconds=elapsed,
        )
