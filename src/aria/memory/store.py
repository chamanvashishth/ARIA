"""Small local conversation-memory store for ARIA."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    role: str
    content: str
    created_at: str
    metadata: dict[str, str]


class LocalMemoryStore:
    """Append-only JSONL memory with bounded retrieval and explicit persistence."""

    ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._entries: list[MemoryEntry] = []
        self._lock = RLock()
        if path is not None and path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        entries: list[MemoryEntry] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    entry = MemoryEntry(
                        id=str(payload["id"]),
                        role=str(payload["role"]),
                        content=str(payload["content"]),
                        created_at=str(payload["created_at"]),
                        metadata={str(k): str(v) for k, v in payload.get("metadata", {}).items()},
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"invalid memory record on line {line_number}") from exc
                if entry.role not in self.ALLOWED_ROLES:
                    raise ValueError(f"invalid memory role on line {line_number}")
                entries.append(entry)
        self._entries = entries

    def add(
        self,
        role: str,
        content: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> MemoryEntry:
        if role not in self.ALLOWED_ROLES:
            raise ValueError(f"unsupported memory role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("memory content must be a non-empty string")
        entry = MemoryEntry(
            id=uuid4().hex,
            role=role,
            content=content,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={str(k): str(v) for k, v in (metadata or {}).items()},
        )
        with self._lock:
            self._entries.append(entry)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    def recent(self, limit: int = 20) -> list[MemoryEntry]:
        if limit < 0:
            raise ValueError("limit cannot be negative")
        with self._lock:
            return list(self._entries[-limit:]) if limit else []

    def search(self, query: str, *, limit: int = 10) -> list[MemoryEntry]:
        """Case-insensitive literal substring search; not semantic retrieval."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        if limit < 0:
            raise ValueError("limit cannot be negative")
        needle = query.casefold()
        with self._lock:
            matches = [entry for entry in reversed(self._entries) if needle in entry.content.casefold()]
        return matches[:limit]

    def clear(self) -> None:
        """Clear memory from both process state and the configured local file."""
        with self._lock:
            self._entries.clear()
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text("", encoding="utf-8")

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
