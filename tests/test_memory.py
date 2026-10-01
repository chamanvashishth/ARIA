from pathlib import Path

import pytest

from aria.memory import LocalMemoryStore


def test_memory_add_recent_and_search() -> None:
    store = LocalMemoryStore()
    first = store.add("user", "I am learning quantum computing")
    store.add("assistant", "Let's start with qubits")
    assert len(store) == 2
    assert store.recent(1)[0].content == "Let's start with qubits"
    assert store.search("QUANTUM")[0].id == first.id


def test_memory_persists_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "memory" / "history.jsonl"
    store = LocalMemoryStore(path)
    store.add("user", "Remember this locally", metadata={"source": "test"})

    restored = LocalMemoryStore(path)
    assert len(restored) == 1
    assert restored.recent()[0].content == "Remember this locally"
    assert restored.recent()[0].metadata == {"source": "test"}


def test_memory_rejects_invalid_role_and_empty_content() -> None:
    store = LocalMemoryStore()
    with pytest.raises(ValueError, match="role"):
        store.add("intruder", "hello")
    with pytest.raises(ValueError, match="non-empty"):
        store.add("user", "   ")


def test_memory_clear_removes_persisted_records(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    store = LocalMemoryStore(path)
    store.add("user", "temporary")
    store.clear()
    assert len(store) == 0
    assert path.read_text(encoding="utf-8") == ""
