"""Local HTTP API boundary for ARIA."""

from aria.api.server import DEFAULT_MAX_BODY_BYTES, create_api_server

__all__ = ["DEFAULT_MAX_BODY_BYTES", "create_api_server"]
