"""Application-level assistant orchestration."""

from aria.interfaces.messages import Message


class Assistant:
    """Coordinate a model responder without coupling to a specific provider."""

    def __init__(self, responder: object) -> None:
        self._responder = responder

    def respond(self, message: Message) -> Message:
        """Return the responder's answer as an assistant message."""
        respond = getattr(self._responder, "respond", None)
        if not callable(respond):
            raise TypeError("responder must expose a callable respond(message) method")

        content = respond(message)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("responder must return non-empty text")

        return Message(role="assistant", content=content)
