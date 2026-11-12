import re
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass

from langchain_core.documents import Document

from ragkb.citations import SourceRef, dedupe_documents, format_sources
from ragkb.config import Settings
from ragkb.prompts import build_messages
from ragkb.store import collection_count

NO_INDEX_MESSAGE = "知识库尚未建库，请先执行 python -m ragkb.ingest"
NO_ANSWER_MESSAGE = "根据现有资料无法回答该问题。"
EMPTY_ANSWER_MESSAGE = (
    "模型未能生成有效回答，请尝试缩短问题或调大 LLM_NUM_PREDICT。"
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[SourceRef]


def extract_text(message) -> str:
    blocks = getattr(message, "content_blocks", None)
    text = ""
    if blocks:
        text = "".join(
            block.get("text", "")
            for block in blocks
            if block.get("type") == "text"
        )
    if not text:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            text = content
    return _THINK_TAG_RE.sub("", text).strip()


class RagService:
    def __init__(self, settings: Settings, vectorstore, llm) -> None:
        self.settings = settings
        self.vectorstore = vectorstore
        self.llm = llm

    def count_chunks(self) -> int:
        return collection_count(self.vectorstore)

    def retrieve(self, question: str) -> list[Document]:
        documents = self.vectorstore.similarity_search(
            question, k=self.settings.retrieval_k
        )
        return dedupe_documents(documents)

    def answer(self, question: str) -> RagAnswer:
        if self.count_chunks() == 0:
            return RagAnswer(NO_INDEX_MESSAGE, [])

        documents = self.retrieve(question)
        if not documents:
            return RagAnswer(NO_ANSWER_MESSAGE, [])

        messages = build_messages(question, documents)
        response = self.llm.invoke(messages)
        text = extract_text(response)
        if not text:
            text = EMPTY_ANSWER_MESSAGE
        return RagAnswer(text, format_sources(documents))

    async def astream_answer(self, question: str) -> AsyncIterator[dict]:
        yield {"event": "status", "data": {"stage": "retrieving"}}

        if self.count_chunks() == 0:
            yield {"event": "error", "data": {"message": NO_INDEX_MESSAGE}}
            return

        documents = self.retrieve(question)
        if not documents:
            yield {"event": "error", "data": {"message": NO_ANSWER_MESSAGE}}
            return

        sources = [asdict(source) for source in format_sources(documents)]
        yield {"event": "sources", "data": {"sources": sources}}
        yield {"event": "status", "data": {"stage": "thinking"}}

        messages = build_messages(question, documents)
        got_text = False
        async for chunk in self.llm.astream(messages):
            for block in getattr(chunk, "content_blocks", None) or []:
                if block.get("type") != "text":
                    continue
                text = block.get("text") or ""
                if not text:
                    continue
                if not got_text:
                    yield {"event": "status", "data": {"stage": "generating"}}
                    got_text = True
                yield {"event": "delta", "data": {"text": text}}

        if not got_text:
            yield {"event": "delta", "data": {"text": EMPTY_ANSWER_MESSAGE}}
        yield {"event": "done", "data": {"ok": True}}
