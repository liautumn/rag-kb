from fastapi.testclient import TestClient

from ragkb.server import app


def test_index_page_and_assets():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "本地 RAG 知识库" in resp.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
