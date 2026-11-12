from dataclasses import dataclass

import httpx
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

from ragkb.config import Settings


@dataclass(frozen=True)
class OllamaStatus:
    ollama_ok: bool
    llm_available: bool
    embed_available: bool


def build_chat_model(settings: Settings) -> ChatOllama | ChatOpenAI:
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "not-needed",
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_completion_tokens=settings.llm_num_predict,
            timeout=settings.request_timeout,
        )
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_num_predict,
        num_ctx=settings.llm_num_ctx,
        reasoning=settings.llm_reasoning,
        client_kwargs={"timeout": settings.request_timeout},
        async_client_kwargs={"timeout": settings.request_timeout},
    )


def build_embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embed_model,
        client_kwargs={"timeout": settings.request_timeout},
    )


def check_ollama(settings: Settings) -> OllamaStatus:
    base = settings.ollama_base_url.rstrip("/")
    try:
        response = httpx.get(f"{base}/api/tags", timeout=3.0)
        response.raise_for_status()
        names = {
            item.get("name", "")
            for item in response.json().get("models", [])
        }
    except Exception:  # noqa: BLE001 - 健康检查任何失败都视为不可用
        return OllamaStatus(False, False, False)

    def available(model: str) -> bool:
        return model in names or f"{model}:latest" in names

    embed_available = available(settings.embed_model)
    if settings.llm_provider == "openai":
        llm_available = _online_llm_available(settings)
    else:
        llm_available = available(settings.llm_model)

    return OllamaStatus(
        ollama_ok=True,
        llm_available=llm_available,
        embed_available=embed_available,
    )


def _online_llm_available(settings: Settings) -> bool:
    base = settings.llm_base_url.rstrip("/")
    try:
        response = httpx.get(
            f"{base}/models",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=3.0,
        )
        response.raise_for_status()
        ids = {
            item.get("id", "")
            for item in response.json().get("data", [])
        }
    except Exception:  # noqa: BLE001 - 在线模型不可用一律视为缺失
        return False
    return settings.llm_model in ids
