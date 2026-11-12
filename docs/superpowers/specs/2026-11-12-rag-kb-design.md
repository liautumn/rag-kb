# 本地 RAG 知识库问答系统设计（rag-kb）

- 日期：2026-11-12
- 状态：设计已获用户确认
- 参考：https://deepseekmodel.com/deepseek-rag#overview

## 1. 背景与目标

在 `/Users/autumn/IdeaProjects/rag-kb` 从零搭建一个完全本地运行的 RAG 知识库问答系统。流程参考 DeepSeek RAG 教程（文档加载 → 切分 → 向量化 → ChromaDB 检索 → Prompt 增强 → 生成 → 来源引用），但生成模型改用 Ollama 本地部署的 `qwen3-vl:8b`，不调用任何云端 API。

目标：

1. 用户把 PDF / TXT / Markdown / DOCX 文档放入 `documents/`，执行一条建库命令即可生成持久化索引。
2. 通过浏览器聊天界面提问，系统基于本地文档回答，并展示来源（文件名、页码、片段）。
3. 除首次拉取 Ollama 模型外，全流程离线运行。

## 2. 现状与约束

| 项目 | 现状 |
|---|---|
| 操作系统 | macOS（darwin），Apple Silicon |
| Python | conda 环境 `rag-kb`，Python 3.13.18，初始只有 pip/setuptools/wheel |
| Ollama | v0.40.16 |
| 生成模型 | `qwen3-vl:8b` 已下载（6.2GB，thinking 版本，支持 vision/tools/thinking，默认 thinking 开启） |
| 嵌入模型 | 选用 `bge-m3`（Ollama，需首次拉取约 1.2GB，嵌入维度 1024） |
| 网络 | PyPI 可用；huggingface.co 直连超时，hf-mirror.com 可用。本方案不依赖 HuggingFace |
| 仓库 | 全新项目，仅有一个初始提交 |

已确认的限制：

- `qwen3-vl:8b` 是纯 thinking 版本，Ollama 的 `think: false` 参数会被忽略（见 ollama/ollama#13353、#14798）。用户选择继续使用该模型，不做额外模型下载。应用层必须隐藏思考内容，并留出足够的输出 token 预算。
- 若未来需要真正关闭思考，可拉取 `qwen3-vl:8b-instruct` 并把 `LLM_MODEL` 改为该名称，代码无需改动。

## 3. 决策记录

| 决策 | 选择 | 备选与放弃原因 |
|---|---|---|
| 生成模型 | 继续用 `qwen3-vl:8b`（thinking） | `8b-instruct` 需再下载 6.2GB；`2b-instruct` 能力弱 |
| 交付形态 | 网页版问答 | CLI 交互、仅一次性脚本 |
| Web 方案 | FastAPI + 原生 HTML/CSS/JS | Streamlit / Gradio：定制性和流式控制较差 |
| 嵌入模型 | Ollama `bge-m3` | BGE-small-zh 需 HuggingFace（直连不可用）；MiniLM 中文效果一般 |
| 向量库 | ChromaDB 持久化 | FAISS（需自管理持久化）、Milvus（过重） |
| 思考处理 | 应用层丢弃 reasoning，前端显示「思考中」 | 修改 RENDERER/PARSER 会把思考混入正文，实测不可用 |

## 4. 非目标

- 不做多用户、登录、权限管理。
- 不做 Web 端文档上传；建库通过 CLI 完成。
- 不做重排序、多查询、父文档检索、混合检索（列为后续扩展）。
- 不调用 DeepSeek 或其他云端 API，不做微调。
- 不对图片/扫描件做 OCR；`qwen3-vl` 的视觉能力不用于入库。
- 不提供 Docker 化部署。

## 5. 技术栈

- Python 3.13（conda `rag-kb`）
- LangChain：`langchain`、`langchain-community`、`langchain-ollama`、`langchain-chroma`、`langchain-text-splitters`
- 向量库：ChromaDB，持久化目录 `./chroma_db`，集合名 `rag_kb`
- LLM：`ChatOllama(model="qwen3-vl:8b")`，走 Ollama 原生 `/api/chat`；思考内容位于 `reasoning` 内容块，只消费 `text` 块
- 嵌入：`OllamaEmbeddings(model="bge-m3")`
- Web：FastAPI + Uvicorn；前端为静态文件，无 Node 构建
- 配置：`python-dotenv` + 环境变量
- 测试：pytest + FastAPI TestClient

## 6. 总体架构

### 6.1 离线建库

```
documents/
   │  按扩展名分发 loader
   ▼
Document（content + metadata: source, page）
   │  RecursiveCharacterTextSplitter
   ▼
Chunk（chunk_size=500, overlap=100，含中文分隔符）
   │  OllamaEmbeddings(bge-m3)
   ▼
ChromaDB（collection=rag_kb，持久化 ./chroma_db）
```

### 6.2 在线问答

```
浏览器 ──POST /api/ask/stream──▶ FastAPI
                                   │ 1. 问题向量化
                                   │ 2. Chroma 相似度检索 Top-k (k=4)
                                   │ 3. 编号来源，拼装中文 Prompt
                                   │ 4. ChatOllama 生成（丢弃 reasoning）
                                   ▼
                     SSE 事件流：status → sources → delta* → done
```

## 7. 模块设计

```
ragkb/
  __init__.py
  config.py      # Settings：从环境变量读取配置并提供默认值
  llm.py         # build_chat_model / build_embeddings / check_ollama
  loaders.py     # load_documents(docs_dir) -> (docs, warnings)
  splitter.py    # build_splitter / split_documents
  store.py       # get_vectorstore / ingest_documents / collection_count / reset_collection
  prompts.py     # SYSTEM_PROMPT / build_context / build_messages
  citations.py   # format_sources：按 (source, page) 去重，生成 150 字预览
  chain.py       # RagService：retrieve / answer / astream_answer
  ingest.py      # 建库 CLI：python -m ragkb.ingest [--rebuild]
  server.py      # FastAPI 应用与路由
  web/           # index.html / style.css / app.js
tests/
  test_config.py
  test_splitter.py
  test_citations.py
  test_loaders.py
  test_prompts.py
  test_api.py
documents/.gitkeep
.env.example
pyproject.toml
README.md
```

### 7.1 模块职责

| 模块 | 职责 | 依赖 |
|---|---|---|
| `config.py` | 读取 `.env`/环境变量，校验数值范围，提供单例 `settings` | 无业务依赖 |
| `llm.py` | 构造 ChatOllama / OllamaEmbeddings；`check_ollama()` 返回服务与模型可用性 | config |
| `loaders.py` | 按扩展名加载 `.pdf/.txt/.md/.docx`，统一元数据；单文件失败只记录警告 | langchain-community |
| `splitter.py` | 构造中文友好切分器并切分文档 | langchain-text-splitters |
| `store.py` | 加载/重建 Chroma；写入文档；统计块数 | langchain-chroma |
| `prompts.py` | 中文系统提示词；把检索结果编号拼成上下文 | 无 |
| `citations.py` | 来源去重、文件名/页码/片段格式化 | 无 |
| `chain.py` | 检索 + 生成；非流式与流式两种接口 | llm/store/prompts |
| `ingest.py` | CLI 入口，加载 → 切分 → 建库，输出统计 | 以上模块 |
| `server.py` | HTTP 接口、SSE、静态页面、健康检查 | chain/config |

## 8. 接口设计

### 8.1 `GET /`

返回 `web/index.html` 聊天页面。

### 8.2 `GET /api/health`

```json
{
  "ollama_ok": true,
  "llm_model": "qwen3-vl:8b",
  "llm_available": true,
  "embed_model": "bge-m3",
  "embed_available": true,
  "indexed_chunks": 128,
  "ready": true
}
```

`ready = ollama_ok && llm_available && embed_available && indexed_chunks > 0`。

### 8.3 `POST /api/ask`

请求：

```json
{"question": "……"}
```

响应：

```json
{
  "answer": "……",
  "sources": [
    {"index": 1, "source": "manual.pdf", "page": 3, "snippet": "……"}
  ]
}
```

### 8.4 `POST /api/ask/stream`

SSE 事件（`text/event-stream`）：

| 事件 | 数据 | 说明 |
|---|---|---|
| `status` | `{"stage": "retrieving" \| "thinking" \| "generating"}` | 前端状态提示 |
| `sources` | 同 `/api/ask` 的 `sources` | 检索完成后立即发送 |
| `delta` | `{"text": "…"}` | 回答增量 |
| `done` | `{"ok": true}` | 正常结束 |
| `error` | `{"message": "…"}` | 可读中文错误 |

## 9. Prompt 设计

系统提示词（中文）：

```
你是企业知识库问答助手。请严格依据下面提供的资料回答问题。

规则：
1. 只使用资料中的信息，不要编造或使用资料之外的知识。
2. 如果资料不足以回答，明确回答“根据现有资料无法回答该问题”。
3. 使用简体中文，回答简洁、专业、条理清晰。
4. 引用资料时在句末标注 [来源X]，X 对应资料编号。
```

上下文格式：

```
[来源1] 文件：manual.pdf（第 3 页）
<chunk 内容>

[来源2] 文件：faq.md
<chunk 内容>
```

用户消息包含 `资料：{context}` 与 `问题：{question}`。

## 10. 思考模式处理

- ChatOllama 返回的内容块中，`type == "reasoning"` 的块全部丢弃，只拼接 `type == "text"`。
- 前端在检索完成后显示「正在思考…」，收到首个 `delta` 后切换为回答输出。
- 默认 `LLM_NUM_PREDICT=4096`、`LLM_NUM_CTX=8192`，避免思考耗尽预算导致空回答。
- 若最终回答为空，返回兜底文案：`模型未能生成有效回答，请尝试缩短问题或调大 LLM_NUM_PREDICT。`
- README 记录真正关闭思考的方法（改用 `qwen3-vl:8b-instruct` 并修改 `LLM_MODEL`）。

## 11. 配置项

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `LLM_MODEL` | `qwen3-vl:8b` | 生成模型 |
| `EMBED_MODEL` | `bge-m3` | 嵌入模型 |
| `DOCS_DIR` | `./documents` | 知识库文档目录 |
| `CHROMA_DIR` | `./chroma_db` | 向量库持久化目录 |
| `COLLECTION_NAME` | `rag_kb` | Chroma 集合名 |
| `CHUNK_SIZE` | `500` | 切分块大小（字符） |
| `CHUNK_OVERLAP` | `100` | 相邻块重叠（字符） |
| `RETRIEVAL_K` | `4` | 检索条数 |
| `LLM_TEMPERATURE` | `0.3` | 生成温度 |
| `LLM_NUM_PREDICT` | `4096` | 最大输出 token |
| `LLM_NUM_CTX` | `8192` | 上下文窗口 token |
| `REQUEST_TIMEOUT` | `300` | 单次生成超时（秒） |

`.env.example` 提供上述全部条目；`.env` 加入 `.gitignore`。

## 12. 错误处理

| 场景 | 行为 |
|---|---|
| Ollama 未启动 | `/api/health` 返回 `ollama_ok=false`；页面顶部红色提示「无法连接 Ollama」 |
| 模型未拉取 | health 标记缺失模型；问答返回 503，中文说明拉取命令 |
| 索引为空/未建库 | 问答返回 400「知识库尚未建库，请先执行 python -m ragkb.ingest」 |
| 文档目录为空 | 建库脚本报错退出（退出码 1） |
| 单个文件解析失败 | 记录 warning，继续处理其余文件；全部失败才报错 |
| 生成超时 | 返回 504「模型响应超时」 |
| 思考耗尽 token 导致空回答 | 返回兜底文案 |
| SSE 中途异常 | 发送 `error` 事件后关闭连接，不抛裸异常 |

## 13. 测试策略

单元测试（不依赖 Ollama，快速）：

- `test_config.py`：默认值、环境变量覆盖、非法数值回退/报错。
- `test_splitter.py`：中文分隔符、块大小与重叠、空文档。
- `test_loaders.py`：临时目录下 TXT/MD 加载、未知扩展名跳过、元数据正确。
- `test_citations.py`：按 (source, page) 去重、页码缺失、预览截断。
- `test_prompts.py`：上下文编号顺序、[来源X] 提示、空检索结果。
- `test_api.py`：用依赖覆盖注入假 `RagService`，验证 `/api/health`、`/api/ask` 正常与错误路径、SSE 事件顺序。

可选集成测试（`RUN_INTEGRATION=1` 时运行，真实调用 Ollama 与 Chroma）：

- 建库 → 提问 → 回答非空且来源非空。

## 14. 验收标准

1. 在 `rag-kb` 环境执行 `pip install -e ".[dev]"` 成功。
2. `ollama pull bge-m3` 成功，`ollama list` 可见。
3. 向 `documents/` 放入至少一份文档后，`python -m ragkb.ingest` 成功并输出块数。
4. `uvicorn ragkb.server:app --port 8000` 启动后，浏览器可完成提问并看到回答与来源。
5. `pytest` 全部通过。
6. `/api/health` 正确反映 Ollama、模型与索引状态。

## 15. 风险与备选

| 风险 | 应对 |
|---|---|
| ChromaDB/onnxruntime 在 Python 3.13 安装失败 | 新建 Python 3.11 conda 环境；或改用 FAISS（需同步调整 store 层） |
| bge-m3 首次拉取慢/中断 | `ollama pull bge-m3` 支持断点续传，重试即可 |
| thinking 版响应慢（约 30–90 秒） | SSE 流式 + 状态提示；可改用 instruct 模型 |
| qwen3-vl:8b 在长上下文下输出不稳定 | 控制检索块数与 Prompt 长度；必要时调低 `RETRIEVAL_K` |
| 中文 PDF 解析质量差 | README 建议优先使用文本型 PDF；扫描件需先 OCR（非目标） |

## 16. 后续可扩展

- 重排序（bge-reranker）、混合检索（向量 + BM25）
- Web 端上传文档与重建索引
- 多轮对话与问题改写
- 引用高亮与原文定位
- Docker Compose 一键部署
