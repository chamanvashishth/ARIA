import json
import pytest

from app import app, safe_calculate


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_safe_calculate_evaluates_basic_arithmetic() -> None:
    assert safe_calculate("2 + 2") == "4"
    assert safe_calculate("10 - 4") == "6"
    assert safe_calculate("6 * 7") == "42"
    assert safe_calculate("100 / 4") == "25"
    assert safe_calculate("(10 + 2) * 3") == "36"
    assert safe_calculate("2 ** 5") == "32"
    assert safe_calculate("10 % 3") == "1"


def test_safe_calculate_guards_against_errors_and_injection() -> None:
    assert "Division by zero" in safe_calculate("1 / 0")
    assert "unsupported" in safe_calculate("__import__('os').system('ls')").lower()
    assert "unsupported" in safe_calculate("open('secrets.txt')").lower()
    assert "empty" in safe_calculate("   ").lower()


def test_index_route_serves_html(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/html")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert b"ARIA // STUDIO" in response.data


def test_status_endpoints(client) -> None:
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "online"
    assert data["intents_count"] > 0
    assert "greeting" in data["available_intents"]

    aria_res = client.get("/api/aria/status")
    assert aria_res.status_code == 200
    aria_data = aria_res.get_json()
    assert aria_data["version"] == "0.1.0"
    assert aria_data["runtime_started"] is True


def test_chat_api_requires_message(client) -> None:
    res = client.post("/api/chat", json={})
    assert res.status_code == 400
    assert "empty" in res.get_json()["error"]


def test_chat_api_evaluates_math(client) -> None:
    res = client.post("/api/chat", json={"message": "calc 12 * 12"})
    assert res.status_code == 200
    data = res.get_json()
    assert "144" in data["reply"]
    assert data["intent"] == "tool_calculate"


def test_chat_api_processes_hybrid_and_rag(client) -> None:
    res = client.post("/api/chat", json={"message": "hello", "engine": "hybrid"})
    assert res.status_code == 200
    assert res.get_json()["intent"] == "greeting"

    rag_res = client.post("/api/chat", json={"message": "ARIA architecture", "engine": "aria_rag"})
    assert rag_res.status_code == 200
    assert "BM25" in rag_res.get_json()["reply"]


def test_chat_stream_endpoint(client) -> None:
    res = client.post("/api/chat/stream", json={"message": "hello"})
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["Content-Type"]
    body = res.get_data(as_text=True)
    assert 'data: {"type": "meta"' in body
    assert 'data: {"type": "done"}' in body


def test_rag_ingest_and_retrain_endpoints(client) -> None:
    ingest_res = client.post(
        "/api/aria/rag/ingest",
        json={"source": "test_doc.txt", "text": "Testing RAG chunk ingestion in ARIA."}
    )
    assert ingest_res.status_code == 200
    assert ingest_res.get_json()["success"] is True

    retrain_res = client.post("/api/retrain")
    assert retrain_res.status_code == 200
    assert retrain_res.get_json()["success"] is True

