from pathlib import Path

import pytest

from ragkb.config import Settings

ENV_NAMES = [
    "OLLAMA_BASE_URL", "LLM_MODEL", "EMBED_MODEL", "DOCS_DIR", "CHROMA_DIR",
    "COLLECTION_NAME", "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_K",
    "LLM_TEMPERATURE", "LLM_NUM_PREDICT", "LLM_NUM_CTX", "REQUEST_TIMEOUT",
    "LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_REASONING",
]


def test_defaults(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    s = Settings.from_env(env_file=None)
    assert s.llm_model == "qwen3-vl:8b"
    assert s.embed_model == "bge-m3"
    assert s.chunk_size == 500
    assert s.chunk_overlap == 100
    assert s.retrieval_k == 4
    assert s.llm_num_predict == 4096
    assert s.request_timeout == 300.0
    assert s.llm_reasoning is True
    assert s.llm_provider == "ollama"


def test_online_provider_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:3425/v1/")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "workbuddy/deepseek-v4.1-flash")
    s = Settings.from_env(env_file=None)
    assert s.llm_provider == "openai"
    assert s.llm_base_url == "http://127.0.0.1:3425/v1"
    assert s.llm_api_key == "test-key"
    assert s.llm_model == "workbuddy/deepseek-v4.1-flash"


def test_openai_provider_requires_base_url():
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        Settings(llm_provider="openai")


def test_invalid_provider():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings(llm_provider="unknown")


def test_llm_reasoning_override(monkeypatch):
    monkeypatch.setenv("LLM_REASONING", "false")
    s = Settings.from_env(env_file=None)
    assert s.llm_reasoning is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen3-vl:8b-instruct")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
    monkeypatch.setenv("DOCS_DIR", "/tmp/docs")
    s = Settings.from_env(env_file=None)
    assert s.llm_model == "qwen3-vl:8b-instruct"
    assert s.chunk_size == 800
    assert s.ollama_base_url == "http://127.0.0.1:11434"
    assert s.docs_dir == Path("/tmp/docs")


def test_invalid_overlap(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings.from_env(env_file=None)


def test_invalid_int(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_K", "abc")
    with pytest.raises(ValueError, match="RETRIEVAL_K"):
        Settings.from_env(env_file=None)
