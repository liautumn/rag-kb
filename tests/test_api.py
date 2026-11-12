import pytest
from fastapi.testclient import TestClient

from ragkb.citations import SourceRef
from ragkb.chain import RagAnswer
from ragkb.llm import OllamaStatus
from ragkb.server import app, get_service


class FakeService:
    def __init__(self, chunks=3):
        self._chunks = chunks

    def count_chunks(self):
        return self._chunks

    def answer(self, question):
        return RagAnswer("答案 [来源1]", [SourceRef(1, "a.txt", None, "片段")])

    async def astream_answer(self, question):
        yield {"event": "status", "data": {"stage": "retrieving"}}
        yield {
            "event": "sources",
            "data": {
                "sources": [
                    {"index": 1, "source": "a.txt", "page": None, "snippet": "片段"}
                ]
            },
        }
        yield {"event": "delta", "data": {"text": "答案"}}
        yield {"event": "done", "data": {"ok": True}}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "ragkb.server.check_ollama",
        lambda settings: OllamaStatus(True, True, True),
    )
    app.dependency_overrides[get_service] = lambda: FakeService()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_ready(client):
    data = client.get("/api/health").json()
    assert data["ready"] is True
    assert data["indexed_chunks"] == 3


def test_health_reports_ollama_down(client, monkeypatch):
    monkeypatch.setattr(
        "ragkb.server.check_ollama",
        lambda settings: OllamaStatus(False, False, False),
    )
    data = client.get("/api/health").json()
    assert data["ollama_ok"] is False
    assert data["ready"] is False


def test_ask_success(client):
    resp = client.post("/api/ask", json={"question": "问题"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "答案 [来源1]"
    assert body["sources"][0]["source"] == "a.txt"


def test_ask_empty_question(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_ask_empty_index_returns_400(client):
    app.dependency_overrides[get_service] = lambda: FakeService(chunks=0)
    resp = client.post("/api/ask", json={"question": "问题"})
    assert resp.status_code == 400
    assert "ragkb.ingest" in resp.json()["detail"]


def test_ask_ollama_down_returns_503(client, monkeypatch):
    monkeypatch.setattr(
        "ragkb.server.check_ollama",
        lambda settings: OllamaStatus(False, False, False),
    )
    resp = client.post("/api/ask", json={"question": "问题"})
    assert resp.status_code == 503


def test_stream_event_order(client):
    resp = client.post("/api/ask/stream", json={"question": "问题"})
    assert resp.status_code == 200
    text = resp.text
    for event in ("status", "sources", "delta", "done"):
        assert f"event: {event}" in text
    assert "答案" in text
