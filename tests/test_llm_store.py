from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from ragkb.config import Settings
from ragkb.llm import OllamaStatus, build_chat_model, check_ollama
from ragkb.store import (
    collection_count,
    get_vectorstore,
    ingest_documents,
    reset_collection,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [1.0, 1.0]


class FakeResponse:
    def __init__(self, names):
        self._names = names

    def raise_for_status(self):
        return None

    def json(self):
        return {"models": [{"name": n} for n in self._names]}


def test_check_ollama_ok(monkeypatch):
    monkeypatch.setattr(
        "ragkb.llm.httpx.get",
        lambda url, timeout: FakeResponse(["qwen3-vl:8b", "bge-m3:latest"]),
    )
    status = check_ollama(Settings())
    assert status == OllamaStatus(True, True, True)


def test_check_ollama_down(monkeypatch):
    def boom(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr("ragkb.llm.httpx.get", boom)
    status = check_ollama(Settings())
    assert status == OllamaStatus(False, False, False)


def test_ollama_name_with_latest_tag(monkeypatch):
    monkeypatch.setattr(
        "ragkb.llm.httpx.get",
        lambda url, timeout: FakeResponse(["qwen3-vl:8b:latest", "bge-m3"]),
    )
    status = check_ollama(Settings())
    assert status.llm_available and status.embed_available


class FakeOpenAIResponse:
    def __init__(self, ids):
        self._ids = ids

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"id": i} for i in self._ids]}


def openai_settings(**kwargs):
    return Settings(
        llm_provider="openai",
        llm_base_url="http://127.0.0.1:3425/v1",
        llm_api_key="test-key",
        llm_model="workbuddy/deepseek-v4.1-flash",
        **kwargs,
    )


def test_build_chat_model_for_online_provider():
    assert isinstance(build_chat_model(openai_settings()), ChatOpenAI)


def test_check_ollama_checks_online_model(monkeypatch):
    def fake_get(url, timeout, headers=None):
        if url.endswith("/api/tags"):
            return FakeResponse(["bge-m3:latest"])
        return FakeOpenAIResponse(["workbuddy/deepseek-v4.1-flash"])

    monkeypatch.setattr("ragkb.llm.httpx.get", fake_get)
    assert check_ollama(openai_settings()) == OllamaStatus(True, True, True)


def test_check_ollama_online_model_missing(monkeypatch):
    def fake_get(url, timeout, headers=None):
        if url.endswith("/api/tags"):
            return FakeResponse(["bge-m3:latest"])
        return FakeOpenAIResponse(["some/other-model"])

    monkeypatch.setattr("ragkb.llm.httpx.get", fake_get)
    status = check_ollama(openai_settings())
    assert status.ollama_ok is True
    assert status.llm_available is False
    assert status.embed_available is True


def test_reset_preserves_collection_identity(tmp_path):
    settings = Settings(chroma_dir=tmp_path / "db", collection_name="test_kb")
    embeddings = FakeEmbeddings()
    docs = [Document(page_content="你好", metadata={"source": "a.txt", "page": None})]
    ingest_documents(docs, settings, embeddings)
    vs = get_vectorstore(settings, embeddings)
    collection_id = vs._collection.id
    reset_collection(settings, embeddings)
    vs2 = get_vectorstore(settings, embeddings)
    assert vs2._collection.id == collection_id
    assert collection_count(vs2) == 0
    settings = Settings(chroma_dir=tmp_path / "db", collection_name="test_kb")
    embeddings = FakeEmbeddings()
    docs = [Document(page_content="你好", metadata={"source": "a.txt", "page": None})]
    assert ingest_documents(docs, settings, embeddings) == 1
    vs = get_vectorstore(settings, embeddings)
    assert collection_count(vs) == 1
    reset_collection(settings, embeddings)
    vs2 = get_vectorstore(settings, embeddings)
    assert collection_count(vs2) == 0
