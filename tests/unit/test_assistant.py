"""Tests for assistant orchestration."""

import pytest

from aria.core.assistant import Assistant
from aria.interfaces.messages import Message


class FakeResponder:
    def respond(self, message: Message) -> str:
        return f"received: {message.content}"


def test_assistant_delegates_to_responder() -> None:
    assistant = Assistant(FakeResponder())
    result = assistant.respond(Message(role="user", content="Hello"))
    assert result == Message(role="assistant", content="received: Hello")


def test_assistant_rejects_invalid_responder() -> None:
    with pytest.raises(TypeError):
        Assistant(object()).respond(Message(role="user", content="Hello"))


def test_assistant_rejects_empty_response() -> None:
    class EmptyResponder:
        def respond(self, message: Message) -> str:
            return " "

    with pytest.raises(ValueError):
        Assistant(EmptyResponder()).respond(Message(role="user", content="Hello"))
