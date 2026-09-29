"""Message contracts used across ARIA components."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Message:
    """A single conversational message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("message role cannot be empty")
        if not self.content.strip():
            raise ValueError("message content cannot be empty")
