from langchain_core.documents import Document

from ragkb.citations import dedupe_documents, format_sources, source_indices


def test_dedupe_by_source_and_page():
    docs = [
        Document(page_content="块一", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="块二", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="块三", metadata={"source": "b.md", "page": None}),
    ]
    result = dedupe_documents(docs)
    assert len(result) == 2
    assert [d.page_content for d in result] == ["块一", "块三"]


def test_dedupe_keeps_distinct_unpaged_chunks():
    docs = [
        Document(page_content="块一", metadata={"source": "a.docx", "page": None}),
        Document(page_content="块二", metadata={"source": "a.docx", "page": None}),
        Document(page_content="块一", metadata={"source": "a.docx", "page": None}),
    ]
    result = dedupe_documents(docs)
    assert [d.page_content for d in result] == ["块一", "块二"]


def test_format_sources_numbers_distinct_unpaged_chunks():
    docs = [
        Document(page_content="第一段", metadata={"source": "a.docx", "page": None}),
        Document(page_content="第二段", metadata={"source": "a.docx", "page": None}),
    ]
    sources = format_sources(docs)
    assert [s.index for s in sources] == [1, 2]
    assert [s.snippet for s in sources] == ["第一段", "第二段"]


def test_source_indices_share_index_for_same_source_page():
    docs = [
        Document(page_content="块一", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="块二", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="块三", metadata={"source": "b.md", "page": None}),
    ]
    assert source_indices(docs) == [1, 1, 2]


def test_format_sources_numbering_and_page():
    docs = [
        Document(page_content="第一段", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="第二段", metadata={"source": "b.md", "page": None}),
    ]
    sources = format_sources(docs)
    assert sources[0].index == 1
    assert sources[0].source == "a.pdf"
    assert sources[0].page == 2
    assert sources[1].index == 2
    assert sources[1].page is None


def test_snippet_truncated_and_whitespace_collapsed():
    docs = [
        Document(
            page_content="  很长的内容  " * 100,
            metadata={"source": "a.txt", "page": None},
        )
    ]
    snippet = format_sources(docs)[0].snippet
    assert len(snippet) == 150
    assert "  " not in snippet


def test_missing_source_defaults():
    docs = [Document(page_content="内容", metadata={})]
    assert format_sources(docs)[0].source == "未知来源"
