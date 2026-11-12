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


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[tuple[str, int | None]] = set()
    result: list[Document] = []
    for document in documents:
        key = _source_and_page(document)
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


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
