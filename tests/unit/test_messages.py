"""Tests for message contracts."""

import pytest

from aria.interfaces.messages import Message


def test_message_accepts_valid_content() -> None:
    message = Message(role="user", content="Hello")
    assert message.role == "user"
    assert message.content == "Hello"


@pytest.mark.parametrize("role, content", [("", "Hello"), ("user", ""), (" ", "Hello")])
def test_message_rejects_empty_values(role: str, content: str) -> None:
    with pytest.raises(ValueError):
        Message(role=role, content=content)
