import pytest
from langchain_core.documents import Document

from ragkb.citations import SourceRef
from ragkb.chain import (
    EMPTY_ANSWER_MESSAGE,
    NO_INDEX_MESSAGE,
    RagAnswer,
    RagService,
    extract_text,
)
from ragkb.config import Settings


class FakeVectorStore:
    def __init__(self, docs, count):
        self._docs = docs
        self._count = count

    def similarity_search(self, query, k):
        return self._docs[:k]

    def get(self):
        return {"ids": [str(i) for i in range(self._count)]}


class FakeMessage:
    def __init__(self, text):
        self.content = text
        self.content_blocks = [{"type": "text", "text": text}]


class FakeChunk:
    def __init__(self, blocks):
        self.content_blocks = blocks


class FakeLLM:
    def __init__(self, text="回答 [来源1]", chunks=None):
        self._text = text
        self._chunks = chunks

    def invoke(self, messages):
        return FakeMessage(self._text)

    async def astream(self, messages):
        for chunk in self._chunks or []:
            yield chunk


def make_service(docs=None, count=1, llm=None):
    docs = docs or [
        Document(page_content="资料", metadata={"source": "a.txt", "page": None})
    ]
    return RagService(Settings(), FakeVectorStore(docs, count), llm or FakeLLM())


def test_extract_text_prefers_text_blocks():
    msg = FakeMessage("你好")
    assert extract_text(msg) == "你好"


def test_answer_returns_text_and_sources():
    service = make_service()
    result = service.answer("问题")
    assert isinstance(result, RagAnswer)
    assert result.answer == "回答 [来源1]"
    assert result.sources == [SourceRef(1, "a.txt", None, "资料")]


def test_retrieve_keeps_all_chunks_for_context():
    docs = [
        Document(page_content="甲", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="乙", metadata={"source": "a.pdf", "page": 1}),
    ]
    service = make_service(docs=docs)
    assert [d.page_content for d in service.retrieve("问题")] == ["甲", "乙"]


def test_extract_text_strips_think_tags():
    msg = FakeMessage("<think>内部思考</think>最终答案")
    assert extract_text(msg) == "最终答案"


def test_empty_index_returns_notice():
    service = make_service(count=0)
    result = service.answer("问题")
    assert result.answer == NO_INDEX_MESSAGE
    assert result.sources == []


def test_empty_model_output_returns_fallback():
    service = make_service(llm=FakeLLM(text=""))
    assert service.answer("问题").answer == EMPTY_ANSWER_MESSAGE


@pytest.mark.asyncio
async def test_astream_discards_reasoning_and_orders_events():
    chunks = [
        FakeChunk([{"type": "reasoning", "reasoning": "内部思考"}]),
        FakeChunk([{"type": "text", "text": "回答"}]),
        FakeChunk([{"type": "text", "text": " [来源1]"}]),
    ]
    service = make_service(llm=FakeLLM(chunks=chunks))
    events = [event async for event in service.astream_answer("问题")]
    names = [e["event"] for e in events]
    assert names[0] == "status"
    assert "sources" in names
    assert names[-1] == "done"
    deltas = "".join(e["data"]["text"] for e in events if e["event"] == "delta")
    assert deltas == "回答 [来源1]"
    assert "内部思考" not in deltas
