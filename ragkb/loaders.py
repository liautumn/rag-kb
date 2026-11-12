from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


@dataclass
class LoadResult:
    documents: list[Document]
    warnings: list[str]


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(f"不支持的文件类型：{path.suffix}")
    return loader.load()


def load_documents(docs_dir: Path) -> LoadResult:
    if not docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在：{docs_dir}")

    documents: list[Document] = []
    warnings: list[str] = []

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(docs_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            warnings.append(f"跳过 {path.name}：不支持的文件类型 {path.suffix}")
            continue
        try:
            loaded = load_file(path)
        except Exception as exc:  # noqa: BLE001 - 单文件失败不中断整批
            warnings.append(f"跳过 {path.name}：{exc}")
            continue

        for doc in loaded:
            metadata = dict(doc.metadata)
            metadata["source"] = relative.as_posix()
            page = metadata.get("page")
            metadata["page"] = page + 1 if isinstance(page, int) else None
            doc.metadata = metadata
        documents.extend(loaded)

    return LoadResult(documents=documents, warnings=warnings)
