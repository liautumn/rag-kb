from langchain_core.documents import Document

from ragkb.splitter import CHINESE_SEPARATORS, split_documents


def test_chinese_text_respects_chunk_size():
    text = "。".join(f"第{i}段内容" * 20 for i in range(20))
    docs = [Document(page_content=text, metadata={"source": "a.txt", "page": None})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 100 for c in chunks)
    assert all(c.metadata["source"] == "a.txt" for c in chunks)


def test_empty_documents():
    assert split_documents([], chunk_size=500, chunk_overlap=100) == []


def test_chinese_separators_present():
    for sep in ("。", "！", "？", "；"):
        assert sep in CHINESE_SEPARATORS
