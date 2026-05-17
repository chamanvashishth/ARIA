"""Response helpers for the ARIA command-line assistant."""

from __future__ import annotations


def get_response(user_message: str, memory: dict | None = None) -> str:
    """Return a simple deterministic response for a user message."""
    normalized = user_message.strip().lower()
    memory = memory or {}

    if not normalized:
        return "Please enter a message so I can help."

    if any(greeting in normalized for greeting in ("hello", "hi", "hey")):
        interactions = int(memory.get("interactions", 0))
        return f"Hello! I'm ARIA. We've had {interactions} saved interaction(s)."

    if "help" in normalized:
        return "I can answer simple prompts, keep a chat history, and store basic memory."

    return f"You said: {user_message}"
