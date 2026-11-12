from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""]

_HEADING_MIN = 2
_HEADING_MAX = 40


def _is_heading(line: str) -> bool:
    content = line.strip()
    if not (_HEADING_MIN <= len(content) <= _HEADING_MAX):
        return False
    if content.startswith("#"):
        return True
    if line.startswith("\t") and "\t" not in line[1:].rstrip("\r\n"):
        return True
    return False


def _heading_offsets(text: str) -> list[int]:
    offsets: list[int] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        if _is_heading(line):
            offsets.append(pos)
        pos += len(line)
    return offsets


def _split_sections(text: str) -> list[str]:
    offsets = _heading_offsets(text)
    if not offsets:
        return [text]
    sections: list[str] = []
    if offsets[0] > 0:
        sections.append(text[: offsets[0]])
    for index, offset in enumerate(offsets):
        end = offsets[index + 1] if index + 1 < len(offsets) else len(text)
        sections.append(text[offset:end])
    return [section for section in sections if section.strip()]


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
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []
    for document in documents:
        for section in _split_sections(document.page_content):
            section_document = Document(
                page_content=section, metadata=document.metadata
            )
            chunks.extend(splitter.split_documents([section_document]))
    return chunks
