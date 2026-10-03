import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from aria.api import create_api_server
from aria.brain import TransformerLanguageModel
from aria.runtime import AriaRuntime


@pytest.fixture
def api():
    model = TransformerLanguageModel(
        vocab_size=260,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        max_sequence_length=32,
        seed=12,
    )
    runtime = AriaRuntime(model)
    runtime.start()
    server = create_api_server(runtime, host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        runtime.stop()


def get_json(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def post_json(url: str, payload: object, content_type: str = "application/json") -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_health_endpoint_reports_runtime_state(api: str) -> None:
    status, payload = get_json(f"{api}/health")
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["runtime_started"] is True


def test_generate_endpoint_returns_structured_result(api: str) -> None:
    status, payload = post_json(
        f"{api}/generate",
        {"prompt": "hi", "config": {"max_new_tokens": 2, "top_k": 3, "seed": 4}},
    )
    assert status == 200
    assert payload["text"].startswith("hi")
    assert payload["prompt_tokens"] == 2
    assert payload["generated_tokens"] == 2
    assert len(payload["token_ids"]) == 4
    assert payload["elapsed_seconds"] >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": "   "},
        {"prompt": "hello", "extra": True},
        {"prompt": "hello", "config": {"max_new_tokens": 0}},
        {"prompt": "hello", "config": {"temperature": 0}},
        {"prompt": "hello", "config": {"unexpected": 1}},
    ],
)
def test_invalid_requests_return_400(api: str, payload: object) -> None:
    status, response = post_json(f"{api}/generate", payload)
    assert status == 400
    assert "error" in response


def test_unknown_route_returns_404(api: str) -> None:
    status, payload = get_json(f"{api}/missing")
    assert status == 404
    assert payload["error"]["code"] == "not_found"


def test_generate_requires_json_content_type(api: str) -> None:
    status, payload = post_json(f"{api}/generate", {"prompt": "hi"}, "text/plain")
    assert status == 415
    assert payload["error"]["code"] == "unsupported_media_type"


def test_server_factory_validates_limits() -> None:
    model = TransformerLanguageModel(
        vocab_size=260,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        max_sequence_length=16,
        seed=12,
    )
    runtime = AriaRuntime(model)
    with pytest.raises(ValueError, match="max_body_bytes"):
        create_api_server(runtime, max_body_bytes=0)
