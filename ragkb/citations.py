from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class SourceRef:
    index: int
    source: str
    page: int | None
    snippet: str


def _source_and_page(document: Document) -> tuple[str, int | None]:
    source = str(document.metadata.get("source") or "未知来源")
    page_raw = document.metadata.get("page")
    page = int(page_raw) if page_raw is not None else None
    return source, page


def _dedupe_key(document: Document) -> tuple[str, int | None, str]:
    source, page = _source_and_page(document)
    if page is not None:
        return (source, page, "")
    return (source, page, document.page_content)


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[tuple[str, int | None, str]] = set()
    result: list[Document] = []
    for document in documents:
        key = _dedupe_key(document)
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


def source_indices(documents: list[Document]) -> list[int]:
    index_by_key: dict[tuple[str, int | None, str], int] = {}
    for document in dedupe_documents(documents):
        index_by_key[_dedupe_key(document)] = len(index_by_key) + 1
    return [index_by_key[_dedupe_key(document)] for document in documents]


def format_sources(
    documents: list[Document], snippet_len: int = 150
) -> list[SourceRef]:
    result: list[SourceRef] = []
    for index, document in enumerate(dedupe_documents(documents), start=1):
        source, page = _source_and_page(document)
        snippet = " ".join(document.page_content.split())[:snippet_len]
        result.append(
            SourceRef(index=index, source=source, page=page, snippet=snippet)
        )
    return result
