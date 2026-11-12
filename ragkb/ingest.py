import argparse

from ragkb.config import get_settings
from ragkb.llm import build_embeddings, check_ollama
from ragkb.loaders import load_documents
from ragkb.splitter import split_documents
from ragkb.store import ingest_documents, reset_collection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="建立本地 RAG 知识库索引")
    parser.add_argument(
        "--rebuild", action="store_true", help="删除现有集合后重建"
    )
    args = parser.parse_args(argv)

    settings = get_settings()

    if not settings.docs_dir.exists():
        print(f"错误：文档目录不存在：{settings.docs_dir}")
        return 1

    result = load_documents(settings.docs_dir)
    for warning in result.warnings:
        print(f"警告：{warning}")
    if not result.documents:
        print(
            "错误：未找到可加载的文档，请把 PDF/TXT/MD/DOCX 放入 "
            f"{settings.docs_dir}"
        )
        return 1

    status = check_ollama(settings)
    if not status.ollama_ok:
        print("错误：无法连接 Ollama，请先启动 ollama serve")
        return 1
    if not status.embed_available:
        print(
            f"错误：未安装嵌入模型 {settings.embed_model}，"
            f"请执行 ollama pull {settings.embed_model}"
        )
        return 1

    chunks = split_documents(
        result.documents, settings.chunk_size, settings.chunk_overlap
    )
    embeddings = build_embeddings(settings)
    if args.rebuild:
        reset_collection(settings, embeddings)

    written = ingest_documents(chunks, settings, embeddings)
    print(
        f"完成：加载 {len(result.documents)} 个文档，"
        f"切分并写入 {written} 个块（集合 {settings.collection_name}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
