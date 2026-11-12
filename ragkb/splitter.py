from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""]


def build_splitter(
    chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
    )


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    return build_splitter(chunk_size, chunk_overlap).split_documents(documents)
