from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else raw


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数，当前值：{raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字，当前值：{raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    return Path(_env_str(name, default))


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    llm_provider: str = "ollama"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen3-vl:8b"
    embed_model: str = "bge-m3"
    docs_dir: Path = Path("documents")
    chroma_dir: Path = Path("chroma_db")
    collection_name: str = "rag_kb"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_k: int = 4
    llm_temperature: float = 0.3
    llm_num_predict: int = 4096
    llm_num_ctx: int = 8192
    llm_reasoning: bool = True
    request_timeout: float = 300.0

    def __post_init__(self) -> None:
        if self.llm_provider not in {"ollama", "openai"}:
            raise ValueError("LLM_PROVIDER 必须是 ollama 或 openai")
        if self.llm_provider == "openai" and not self.llm_base_url:
            raise ValueError("LLM_PROVIDER=openai 时必须设置 LLM_BASE_URL")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE 必须大于 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        if self.retrieval_k <= 0:
            raise ValueError("RETRIEVAL_K 必须大于 0")
        if self.llm_num_predict <= 0:
            raise ValueError("LLM_NUM_PREDICT 必须大于 0")
        if self.llm_num_ctx <= 0:
            raise ValueError("LLM_NUM_CTX 必须大于 0")
        if self.request_timeout <= 0:
            raise ValueError("REQUEST_TIMEOUT 必须大于 0")
        if self.llm_temperature < 0:
            raise ValueError("LLM_TEMPERATURE 必须大于等于 0")

    @classmethod
    def from_env(cls, env_file: Path | None = Path(".env")) -> "Settings":
        if env_file is not None and Path(env_file).exists():
            load_dotenv(env_file, override=False)
        return cls(
            ollama_base_url=_env_str(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            llm_provider=_env_str("LLM_PROVIDER", "ollama").strip().lower(),
            llm_base_url=_env_str("LLM_BASE_URL", "").rstrip("/"),
            llm_api_key=_env_str(
                "LLM_API_KEY", _env_str("ANTHROPIC_AUTH_TOKEN", "")
            ),
            llm_model=_env_str("LLM_MODEL", "qwen3-vl:8b"),
            embed_model=_env_str("EMBED_MODEL", "bge-m3"),
            docs_dir=_env_path("DOCS_DIR", "documents"),
            chroma_dir=_env_path("CHROMA_DIR", "chroma_db"),
            collection_name=_env_str("COLLECTION_NAME", "rag_kb"),
            chunk_size=_env_int("CHUNK_SIZE", 500),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 100),
            retrieval_k=_env_int("RETRIEVAL_K", 4),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.3),
            llm_num_predict=_env_int("LLM_NUM_PREDICT", 4096),
            llm_num_ctx=_env_int("LLM_NUM_CTX", 8192),
            llm_reasoning=_env_bool("LLM_REASONING", True),
            request_timeout=_env_float("REQUEST_TIMEOUT", 300.0),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
