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


def test_section_heading_starts_a_chunk():
    text = "\t读 EPC 标签\n\n本指令用于配置读写器。\n\n" + "参数内容。" * 300
    docs = [Document(page_content=text, metadata={"source": "a.docx", "page": None})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=40)
    assert any(c.page_content.lstrip().startswith("读 EPC 标签") for c in chunks)
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["source"] == "a.docx" for c in chunks)
