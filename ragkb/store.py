from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragkb.config import Settings


def get_vectorstore(settings: Settings, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def ingest_documents(
    documents: list[Document], settings: Settings, embeddings: Embeddings
) -> int:
    vectorstore = get_vectorstore(settings, embeddings)
    vectorstore.add_documents(documents)
    return len(documents)


def collection_count(vectorstore: Chroma) -> int:
    collection = getattr(vectorstore, "_collection", None)
    if collection is not None:
        return collection.count()
    return len(vectorstore.get().get("ids", []))


def reset_collection(settings: Settings, embeddings: Embeddings) -> None:
    vectorstore = get_vectorstore(settings, embeddings)
    if collection_count(vectorstore) > 0:
        vectorstore.delete_collection()
