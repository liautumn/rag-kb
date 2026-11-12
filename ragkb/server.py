import asyncio
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragkb.chain import NO_INDEX_MESSAGE, RagService
from ragkb.config import Settings, get_settings
from ragkb.llm import (
    build_chat_model,
    build_embeddings,
    check_ollama,
)
from ragkb.store import get_vectorstore

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="rag-kb")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class HealthResponse(BaseModel):
    ollama_ok: bool
    llm_model: str
    llm_available: bool
    embed_model: str
    embed_available: bool
    indexed_chunks: int
    ready: bool


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceItem(BaseModel):
    index: int
    source: str
    page: int | None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@lru_cache(maxsize=1)
def get_service() -> RagService:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    vectorstore = get_vectorstore(settings, embeddings)
    llm = build_chat_model(settings)
    return RagService(settings, vectorstore, llm)


def _ensure_ready(service: RagService) -> Settings:
    settings = get_settings()
    status = check_ollama(settings)
    if not status.ollama_ok:
        raise HTTPException(
            status_code=503, detail="无法连接 Ollama，请先运行 ollama serve"
        )
    if not status.llm_available:
        raise HTTPException(
            status_code=503,
            detail=f"生成模型未安装，请执行 ollama pull {settings.llm_model}",
        )
    if not status.embed_available:
        raise HTTPException(
            status_code=503,
            detail=f"嵌入模型未安装，请执行 ollama pull {settings.embed_model}",
        )
    if service.count_chunks() == 0:
        raise HTTPException(status_code=400, detail=NO_INDEX_MESSAGE)
    return settings


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health(service: RagService = Depends(get_service)) -> HealthResponse:
    settings = get_settings()
    status = check_ollama(settings)
    try:
        indexed_chunks = service.count_chunks()
    except Exception:  # noqa: BLE001 - 健康检查不应因索引异常而 500
        indexed_chunks = 0
    return HealthResponse(
        ollama_ok=status.ollama_ok,
        llm_model=settings.llm_model,
        llm_available=status.llm_available,
        embed_model=settings.embed_model,
        embed_available=status.embed_available,
        indexed_chunks=indexed_chunks,
        ready=(
            status.ollama_ok
            and status.llm_available
            and status.embed_available
            and indexed_chunks > 0
        ),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(
    request: AskRequest, service: RagService = Depends(get_service)
) -> AskResponse:
    _ensure_ready(service)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")
    try:
        result = service.answer(question)
    except HTTPException:
        raise
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="模型响应超时，请稍后重试"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - 对外返回可读中文错误
        raise HTTPException(status_code=503, detail=f"生成失败：{exc}") from exc
    return AskResponse(
        answer=result.answer,
        sources=[SourceItem(**asdict(source)) for source in result.sources],
    )


@app.post("/api/ask/stream")
async def ask_stream(
    request: AskRequest, service: RagService = Depends(get_service)
):
    settings = _ensure_ready(service)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")

    async def event_generator():
        try:
            async with asyncio.timeout(settings.request_timeout):
                async for event in service.astream_answer(question):
                    yield _sse(event["event"], event["data"])
        except TimeoutError:
            yield _sse("error", {"message": "模型响应超时，请稍后重试"})
        except Exception as exc:  # noqa: BLE001 - SSE 内异常转成 error 事件
            yield _sse("error", {"message": f"生成失败：{exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
