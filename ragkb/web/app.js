const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const send = document.getElementById("send");
const messages = document.getElementById("messages");
const statusEl = document.getElementById("status");
const statusText = statusEl.querySelector(".status__text");
const banner = document.getElementById("banner");

const STAGE_LABELS = {
  retrieving: "正在检索相关段落…",
  thinking: "正在思考…",
  generating: "正在写回答…",
};

let busy = false;

function setStatus(state, text) {
  statusEl.dataset.state = state;
  statusText.textContent = text;
}

function showBanner(text) {
  banner.textContent = text;
  banner.hidden = false;
}

function hideBanner() {
  banner.hidden = true;
  banner.textContent = "";
}

function setBusy(next) {
  busy = next;
  send.disabled = next;
  send.setAttribute("aria-busy", String(next));
}

function scrollToEnd() {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.scrollTo({
    top: document.body.scrollHeight,
    behavior: reduced ? "auto" : "smooth",
  });
}

function healthMessage(health) {
  if (!health.ollama_ok) {
    return "无法连接 Ollama。请先运行 ollama serve。";
  }
  if (!health.llm_available) {
    return `生成模型未安装。请执行：ollama pull ${health.llm_model}`;
  }
  if (!health.embed_available) {
    return `嵌入模型未安装。请执行：ollama pull ${health.embed_model}`;
  }
  if (health.indexed_chunks === 0) {
    return "知识库尚未建库。请先执行：python -m ragkb.ingest";
  }
  return "知识库未就绪。";
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("health request failed");
    }
    const health = await response.json();
    if (health.ready) {
      setStatus("ready", `知识库就绪（${health.indexed_chunks} 个块）`);
      hideBanner();
    } else {
      setStatus("bad", "尚未就绪");
      showBanner(healthMessage(health));
    }
  } catch {
    setStatus("bad", "服务不可用");
    showBanner(
      "无法连接本地服务。请确认 uvicorn 已启动：uvicorn ragkb.server:app --port 8000"
    );
  }
}

function addTurn(role, text = "") {
  const turn = document.createElement("section");
  turn.className = `turn turn--${role}`;

  const node = document.createElement("span");
  node.className = "turn__node";
  node.setAttribute("aria-hidden", "true");

  const body = document.createElement("div");
  body.className = "turn__body";

  const label = document.createElement("p");
  label.className = "turn__label";
  label.textContent = role === "user" ? "你" : "知识库";

  const content = document.createElement("p");
  content.className = "turn__text";
  content.textContent = text;

  body.append(label, content);
  turn.append(node, body);
  messages.append(turn);
  scrollToEnd();
  return { turn, body, content };
}

function addStatus(body, text) {
  const status = document.createElement("p");
  status.className = "turn__status";
  status.textContent = text;
  body.append(status);
  return status;
}

function renderSources(body, sources) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return;
  }

  const box = document.createElement("div");
  box.className = "sources";

  const title = document.createElement("p");
  title.className = "sources__title";
  title.textContent = `来源（${sources.length}）`;
  box.append(title);

  for (const source of sources) {
    const item = document.createElement("article");
    item.className = "source";

    const index = document.createElement("span");
    index.className = "source__index";
    index.textContent = `[${source.index}]`;

    const content = document.createElement("div");

    const meta = document.createElement("p");
    meta.className = "source__meta";
    meta.textContent = source.page
      ? `${source.source}，第 ${source.page} 页`
      : source.source;

    const snippet = document.createElement("p");
    snippet.className = "source__snippet";
    snippet.textContent = source.snippet;

    content.append(meta, snippet);
    item.append(index, content);
    box.append(item);
  }

  body.append(box);
  scrollToEnd();
}

function parseEvent(raw) {
  let name = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      name = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  try {
    return { name, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

async function readStream(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseEvent(raw);
      if (event) {
        onEvent(event.name, event.data);
      }
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    const event = parseEvent(buffer);
    if (event) {
      onEvent(event.name, event.data);
    }
  }
}

async function ask(question) {
  if (busy) {
    return;
  }

  setBusy(true);
  hideBanner();

  const welcome = messages.querySelector('[data-empty="true"]');
  if (welcome) {
    welcome.remove();
  }

  addTurn("user", question);

  const assistant = addTurn("assistant");
  assistant.content.hidden = true;
  const statusLine = addStatus(assistant.body, STAGE_LABELS.retrieving);
  let gotText = false;

  function handleEvent(name, data) {
    if (name === "status") {
      if (statusLine.isConnected) {
        statusLine.textContent = STAGE_LABELS[data.stage] || "处理中…";
      }
      return;
    }

    if (name === "sources") {
      renderSources(assistant.body, data.sources || []);
      return;
    }

    if (name === "delta") {
      if (!gotText) {
        gotText = true;
        statusLine.remove();
        assistant.content.hidden = false;
      }
      assistant.content.textContent += data.text || "";
      scrollToEnd();
      return;
    }

    if (name === "error") {
      statusLine.remove();
      assistant.content.hidden = false;
      assistant.content.textContent = data.message || "生成失败，请稍后重试。";
      assistant.turn.classList.add("turn--error");
      return;
    }

    if (name === "done") {
      if (statusLine.isConnected) {
        statusLine.remove();
      }
      if (!gotText) {
        assistant.content.hidden = false;
        assistant.content.textContent =
          "模型没有返回内容。请重试，或调大 LLM_NUM_PREDICT。";
      }
    }
  }

  try {
    const response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      let detail = `请求失败（${response.status}）。`;
      try {
        const payload = await response.json();
        if (payload.detail) {
          detail = payload.detail;
        }
      } catch {
        // 保持默认错误信息
      }
      throw new Error(detail);
    }

    if (!response.body) {
      throw new Error("当前浏览器不支持流式响应。");
    }

    await readStream(response.body, handleEvent);
  } catch (error) {
    if (statusLine.isConnected) {
      statusLine.remove();
    }
    assistant.content.hidden = false;
    assistant.content.textContent = "请求失败，请稍后重试。";
    assistant.turn.classList.add("turn--error");
    showBanner(error.message || "请求失败，请稍后重试。");
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || busy) {
    return;
  }
  input.value = "";
  input.style.height = "auto";
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.isComposing || event.keyCode === 229) {
    return;
  }
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
});

checkHealth();
