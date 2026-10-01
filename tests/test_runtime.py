import pytest

from aria.brain import TransformerLanguageModel
from aria.memory import LocalMemoryStore
from aria.runtime import AriaRuntime, RuntimeConfig


def make_runtime() -> AriaRuntime:
    model = TransformerLanguageModel(
        vocab_size=260,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        max_sequence_length=16,
        seed=12,
    )
    return AriaRuntime(model)


def test_runtime_requires_start() -> None:
    runtime = make_runtime()
    with pytest.raises(RuntimeError, match="not started"):
        runtime.generate("hi")


def test_runtime_lifecycle_and_generation() -> None:
    runtime = make_runtime()
    assert not runtime.started
    runtime.start()
    assert runtime.started

    result = runtime.generate(
        "hi",
        RuntimeConfig(max_new_tokens=2, top_k=3, seed=4),
    )
    assert result.prompt_tokens == 2
    assert result.generated_tokens == 2
    assert len(result.token_ids) == 4
    assert result.text.startswith("hi")
    assert result.elapsed_seconds >= 0.0

    runtime.stop()
    assert not runtime.started


def test_runtime_rejects_empty_prompt() -> None:
    runtime = make_runtime()
    runtime.start()
    with pytest.raises(ValueError, match="non-empty"):
        runtime.generate("")


def test_runtime_records_turn_when_memory_is_configured() -> None:
    memory = LocalMemoryStore()
    runtime = AriaRuntime(make_runtime().model, memory=memory)
    runtime.start()
    runtime.generate("hi", RuntimeConfig(max_new_tokens=1, top_k=1, seed=2))
    entries = memory.recent()
    assert [entry.role for entry in entries] == ["user", "assistant"]
    assert entries[0].content == "hi"
