# rag-kb

完全本地运行的 RAG 知识库问答系统：把本机文档切分、向量化后存入 ChromaDB，提问时检索相关段落，交给 Ollama 本地模型生成带来源引用的回答。

- 生成模型：可配置为本地 Ollama（如 `qwen3-vl:8b`）或 OpenAI 兼容在线接口（如 magpie 代理）
- 嵌入模型：本地 Ollama `bge-m3`（不依赖云服务）
- 向量库：ChromaDB（持久化到 `./chroma_db`）
- Web：FastAPI + 原生 HTML/CSS/JS，无 Node 构建、无外部 CDN
- 隐私：嵌入与向量检索完全本地；使用在线生成模型时，问题与检索到的资料片段会发送到所配置的在线接口

## 工作方式

```text
离线建库
documents/ ──加载──▶ Document ──切分──▶ Chunk ──bge-m3──▶ ChromaDB

在线问答
浏览器 ──提问──▶ FastAPI ──检索 Top4──▶ 拼装中文 Prompt
                                              │
                                              ▼
                        生成模型（本地 Ollama 或在线 OpenAI 兼容接口）
                                              │
                              回答 + 来源 ──SSE 流式──▶ 浏览器
```

## 环境要求

- macOS / Linux（已在 Apple Silicon + macOS 验证）
- conda 环境 `rag-kb`，Python 3.11+
- Ollama 已安装并运行（用于本地嵌入模型 `bge-m3`）
- 本地生成模型（可选）：`ollama pull qwen3-vl:8b`
- 在线生成模型（可选）：OpenAI 兼容接口的地址、模型名与 API Key

## 安装

```bash
conda activate rag-kb
python -m pip install -e ".[dev]"
ollama pull bge-m3
cp .env.example .env
```

> 国内网络如果安装依赖较慢，可加镜像：
> `python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 建立知识库

1. 把 PDF / TXT / Markdown / DOCX 文档放进 `documents/`（支持子目录）。
2. 执行建库：

```bash
python -m ragkb.ingest --rebuild
```

输出示例：

```text
完成：加载 3 个文档，切分并写入 126 个块（集合 rag_kb）
```

`--rebuild` 会先删除旧集合；不加则增量追加。单个文件解析失败会打印警告并跳过，不中断整批。

## 启动网页

```bash
uvicorn ragkb.server:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。页面顶部会显示知识库状态；未就绪时会给出具体修复命令。

## 配置

配置来自环境变量或项目根目录的 `.env`，完整示例见 `.env.example`。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 地址（嵌入模型使用） |
| `LLM_PROVIDER` | `ollama` | 生成模型提供方：`ollama` 或 `openai` |
| `LLM_BASE_URL` | 空 | 在线接口地址，例如 `http://127.0.0.1:3425/v1` |
| `LLM_API_KEY` | 空 | 在线接口密钥；留空时复用 `ANTHROPIC_AUTH_TOKEN` |
| `LLM_MODEL` | `qwen3-vl:8b` | 生成模型名；在线示例 `workbuddy/deepseek-v4.1-flash` |
| `EMBED_MODEL` | `bge-m3` | 嵌入模型（始终走本地 Ollama） |
| `DOCS_DIR` | `./documents` | 文档目录 |
| `CHROMA_DIR` | `./chroma_db` | 向量库目录 |
| `COLLECTION_NAME` | `rag_kb` | Chroma 集合名 |
| `CHUNK_SIZE` | `500` | 切分块大小（字符） |
| `CHUNK_OVERLAP` | `100` | 相邻块重叠（字符） |
| `RETRIEVAL_K` | `4` | 检索条数 |
| `LLM_TEMPERATURE` | `0.3` | 生成温度 |
| `LLM_NUM_PREDICT` | `4096` | 最大输出 token |
| `LLM_NUM_CTX` | `8192` | 上下文窗口 |
| `LLM_REASONING` | `true` | 是否让模型输出思考内容（`qwen3-vl:8b` 需保持 `true`） |
| `REQUEST_TIMEOUT` | `300` | 单次生成超时（秒） |

## HTTP 接口

- `GET /api/health`：Ollama 连通性、模型是否安装、已索引块数
- `POST /api/ask`：非流式问答，返回 `{answer, sources}`
- `POST /api/ask/stream`：SSE 流式问答，事件为 `status`、`sources`、`delta`、`done`、`error`

```bash
curl -s -X POST http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"知识库构建命令是什么？"}'
```

## 使用在线生成模型（OpenAI 兼容）

生成模型与嵌入模型是解耦的：生成可以走任意 OpenAI 兼容接口，嵌入仍用本地 `bge-m3`。
以 magpie 代理为例，`.env` 配置：

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=http://127.0.0.1:3425/v1
LLM_MODEL=workbuddy/deepseek-v4.1-flash
LLM_API_KEY=...        # 留空时自动复用 ANTHROPIC_AUTH_TOKEN

OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=bge-m3
```

重启服务即可生效。换生成模型不需要重建索引；只有换嵌入模型时才需要重新建库。
在线模型不使用 `LLM_REASONING`，是否思考由服务端决定。

## 关于思考模式

> 以下说明仅适用于 `LLM_PROVIDER=ollama` 的本地生成模型。

`qwen3-vl:8b` 是 thinking 版本，Ollama 会忽略 `think: false`（见
[ollama#13353](https://github.com/ollama/ollama/issues/13353)、
[#14798](https://github.com/ollama/ollama/issues/14798)）。本项目采用：

- `ChatOllama(reasoning=True)` 把思考内容隔离到 `reasoning_content`；
- 服务端只消费正文文本，丢弃思考内容；
- 页面在等待时显示「正在思考…」，不会显示思考文本。

因此响应会明显慢于非思考模型。若想真正关闭思考：

```bash
ollama pull qwen3-vl:8b-instruct
# 把 .env 中的 LLM_MODEL 改为 qwen3-vl:8b-instruct
# 并把 LLM_REASONING 改为 false，然后重启服务
```

代码不需要改动。

## 测试

```bash
python -m pytest
```

单元测试使用假模型/假向量库，不依赖 Ollama 或在线接口。当前共 54 个测试，覆盖配置、切分、加载、来源去重、Prompt、Chroma 写入、RAG 服务、CLI、HTTP/SSE 与页面资源。

## 常见问题

**问答返回「知识库尚未建库」**
先执行 `python -m ragkb.ingest --rebuild`。

**页面提示无法连接 Ollama**
运行 `ollama serve`，或检查 `.env` 中的 `OLLAMA_BASE_URL`。

**提示嵌入模型未安装**
执行 `ollama pull bge-m3`。

**检索效果不好**
优先检查文档是否为可复制文本；扫描版 PDF 需要先 OCR（本项目不做 OCR）。也可以调整 `CHUNK_SIZE`、`CHUNK_OVERLAP`、`RETRIEVAL_K` 后重建。

**ChromaDB 在 Python 3.13 安装失败**
新建 Python 3.11 的 conda 环境后重新安装依赖。

**HuggingFace 无法访问**
本项目不需要 HuggingFace，嵌入全部走 Ollama。

## 目录结构

```text
ragkb/
  config.py     配置
  loaders.py    多格式文档加载
  splitter.py   中文友好切分
  citations.py  来源去重与格式化
  prompts.py    Prompt 与上下文拼装
  llm.py        Ollama 模型工厂与健康检查
  store.py      Chroma 建库/计数/重置
  chain.py      检索 + 生成（含流式）
  ingest.py     建库 CLI
  server.py     FastAPI + SSE
  web/          聊天页面
tests/          单元测试
documents/      知识库文档
```
