from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = """你是企业知识库问答助手。请严格依据下面提供的资料回答问题。

规则：
1. 只使用资料中的信息，不要编造或使用资料之外的知识。
2. 如果资料不足以回答，明确回答“根据现有资料无法回答该问题”。
3. 使用简体中文，回答简洁、专业、条理清晰。
4. 引用资料时在句末标注 [来源X]，X 对应资料编号。
"""

NO_CONTEXT_TEXT = "（无相关资料）"


def build_context(documents: list[Document]) -> str:
    if not documents:
        return NO_CONTEXT_TEXT

    parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source") or "未知来源"
        page = document.metadata.get("page")
        header = f"[来源{index}] 文件：{source}"
        if page is not None:
            header += f"（第 {page} 页）"
        parts.append(f"{header}\n{document.page_content.strip()}")
    return "\n\n".join(parts)


def build_messages(question: str, documents: list[Document]) -> list[BaseMessage]:
    context = build_context(documents)
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"资料：\n{context}\n\n问题：{question}"),
    ]
