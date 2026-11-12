from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from ragkb.prompts import NO_CONTEXT_TEXT, build_context, build_messages


def test_context_numbering_and_page_header():
    docs = [
        Document(page_content="甲内容", metadata={"source": "a.pdf", "page": 3}),
        Document(page_content="乙内容", metadata={"source": "b.md", "page": None}),
    ]
    context = build_context(docs)
    assert "[来源1] 文件：a.pdf（第 3 页）\n甲内容" in context
    assert "[来源2] 文件：b.md\n乙内容" in context


def test_context_keeps_distinct_unpaged_chunks():
    docs = [
        Document(page_content="甲内容", metadata={"source": "a.docx", "page": None}),
        Document(page_content="乙内容", metadata={"source": "a.docx", "page": None}),
    ]
    context = build_context(docs)
    assert "[来源1] 文件：a.docx\n甲内容" in context
    assert "[来源2] 文件：a.docx\n乙内容" in context


def test_context_groups_same_source_page_chunks_under_one_index():
    docs = [
        Document(page_content="甲内容", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="乙内容", metadata={"source": "a.pdf", "page": 2}),
    ]
    context = build_context(docs)
    assert "[来源1] 文件：a.pdf（第 2 页）\n甲内容" in context
    assert "[来源1] 文件：a.pdf（第 2 页）\n乙内容" in context
    assert "[来源2]" not in context


def test_build_messages_contains_question_and_context():
    docs = [Document(page_content="资料内容", metadata={"source": "a.txt", "page": None})]
    messages = build_messages("问题是什么？", docs)
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "问题是什么？" in messages[1].content
    assert "资料内容" in messages[1].content
    assert "[来源1]" in messages[1].content


def test_empty_documents_use_placeholder():
    assert NO_CONTEXT_TEXT in build_context([])
    messages = build_messages("问题", [])
    assert NO_CONTEXT_TEXT in messages[1].content
