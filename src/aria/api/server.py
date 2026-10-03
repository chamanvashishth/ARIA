"""Dependency-free local HTTP API for an injected ARIA runtime."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from aria.runtime import AriaRuntime, RuntimeConfig


DEFAULT_MAX_BODY_BYTES = 64 * 1024


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _error(message: str, code: str) -> dict[str, str]:
    return {"error": {"code": code, "message": message}}


def _validate_generation_request(payload: Any) -> tuple[str, RuntimeConfig]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    unexpected = sorted(set(payload) - {"prompt", "config"})
    if unexpected:
        raise ValueError(f"unexpected field(s): {', '.join(unexpected)}")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' must be a non-empty string")

    raw_config = payload.get("config", {})
    if not isinstance(raw_config, dict):
        raise ValueError("'config' must be a JSON object")
    allowed = {"max_new_tokens", "temperature", "top_k", "seed"}
    extras = sorted(set(raw_config) - allowed)
    if extras:
        raise ValueError(f"unexpected config field(s): {', '.join(extras)}")

    max_new_tokens = raw_config.get("max_new_tokens", 32)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 1024:
        raise ValueError("'max_new_tokens' must be an integer between 1 and 1024")

    temperature = raw_config.get("temperature", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 < temperature <= 5:
        raise ValueError("'temperature' must be a number greater than 0 and at most 5")

    top_k = raw_config.get("top_k")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError("'top_k' must be a positive integer or null")

    seed = raw_config.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("'seed' must be an integer or null")

    return prompt, RuntimeConfig(
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        top_k=top_k,
        seed=seed,
    )


def create_api_server(
    runtime: AriaRuntime,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> ThreadingHTTPServer:
    """Create a local API server around an existing runtime.

    The runtime must already be started. Loopback is the default to avoid
    exposing an unauthenticated generation endpoint to other machines.
    """
    if not isinstance(runtime, AriaRuntime):
        raise TypeError("runtime must be an AriaRuntime instance")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int) or max_body_bytes < 1:
        raise ValueError("max_body_bytes must be a positive integer")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ARIA-Local-API"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:
            # Avoid writing prompt text or request data to standard logs.
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "runtime_started": runtime.started,
                    "service": "aria-local-api",
                })
                return
            self._send_json(404, _error("route not found", "not_found"))

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path != "/generate":
                self._send_json(404, _error("route not found", "not_found"))
                return

            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                self._send_json(415, _error("Content-Type must be application/json", "unsupported_media_type"))
                return

            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length) if raw_length is not None else -1
            except ValueError:
                length = -1
            if length < 0:
                self._send_json(411, _error("a valid Content-Length header is required", "length_required"))
                return
            if length > max_body_bytes:
                self._send_json(413, _error("request body is too large", "body_too_large"))
                return

            try:
                raw_body = self.rfile.read(length)
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, _error("request body must contain valid UTF-8 JSON", "invalid_json"))
                return

            try:
                prompt, config = _validate_generation_request(payload)
            except ValueError as exc:
                self._send_json(400, _error(str(exc), "invalid_request"))
                return

            if not runtime.started:
                self._send_json(503, _error("ARIA runtime is not started", "runtime_unavailable"))
                return

            try:
                result = runtime.generate(prompt, config)
            except (ValueError, RuntimeError) as exc:
                self._send_json(400, _error(str(exc), "generation_rejected"))
                return
            except Exception:
                self._send_json(500, _error("generation failed", "generation_error"))
                return

            self._send_json(200, {
                "text": result.text,
                "token_ids": result.token_ids,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "elapsed_seconds": result.elapsed_seconds,
            })

        def do_OPTIONS(self) -> None:
            self._send_json(405, _error("method not allowed", "method_not_allowed"))

    return ThreadingHTTPServer((host, port), Handler)
