let currentConversationId = null;
let isCodingMode = true;
let socket = null;

document.addEventListener("DOMContentLoaded", () => {
  initUI();
  initWebSocket();
  loadConversations();
  loadProjects();
});

function initUI() {
  const toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
  const sidebar = document.getElementById("sidebar");
  toggleSidebarBtn.addEventListener("click", () => {
    sidebar.classList.toggle("hidden");
  });

  const toggleEditorBtn = document.getElementById("toggleEditorBtn");
  const closeEditorBtn = document.getElementById("closeEditorBtn");
  const editorPane = document.getElementById("editorPane");

  toggleEditorBtn.addEventListener("click", () => {
    editorPane.classList.toggle("hidden");
  });
  closeEditorBtn.addEventListener("click", () => {
    editorPane.classList.add("hidden");
  });

  const modeToggleBtn = document.getElementById("modeToggleBtn");
  const modeBadge = document.getElementById("modeBadge");
  modeToggleBtn.addEventListener("click", () => {
    isCodingMode = !isCodingMode;
    if (isCodingMode) {
      modeToggleBtn.innerHTML = `<i class="fa-solid fa-code"></i> Coding Mode: ON`;
      modeBadge.textContent = "CODING MODE";
      modeBadge.className = "mode-badge coding";
    } else {
      modeToggleBtn.innerHTML = `<i class="fa-solid fa-comments"></i> Chat Mode: ON`;
      modeBadge.textContent = "CHAT MODE";
      modeBadge.className = "mode-badge";
    }
  });

  document.getElementById("newChatBtn").addEventListener("click", createNewConversation);
  document.getElementById("sendBtn").addEventListener("click", sendMessage);

  const promptInput = document.getElementById("promptInput");
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  document.getElementById("runCodeBtn").addEventListener("click", runEditorCode);
  document.getElementById("analyzeCodeBtn").addEventListener("click", analyzeEditorCode);
}

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log("WebSocket connected to CodeMind AI");
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.event === "step_update") {
      if (data.chunk) {
        appendStreamChunk(data.chunk);
      }
    } else if (data.event === "done") {
      finalizeStreamMessage();
    }
  };

  socket.onerror = (err) => {
    console.warn("WebSocket error, falling back to local handler: ", err);
  };
}

async function createNewConversation() {
  const model = document.getElementById("modelSelect").value;
  try {
    const resp = await fetch("/api/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: isCodingMode ? "New Local Coding Task" : "New Chat",
        model: model,
        mode: isCodingMode ? "coding" : "chat"
      })
    });
    const conv = await resp.json();
    currentConversationId = conv.id;
    document.getElementById("chatTitle").textContent = conv.title;
    document.getElementById("chatMessages").innerHTML = "";
    loadConversations();
  } catch (err) {
    console.error("Failed creating conversation:", err);
  }
}

async function loadConversations() {
  try {
    const resp = await fetch("/api/chats");
    const convs = await resp.json();
    const listEl = document.getElementById("conversationsList");
    listEl.innerHTML = "";

    convs.forEach(c => {
      const item = document.createElement("div");
      item.className = `nav-item ${c.id === currentConversationId ? "active" : ""}`;
      item.innerHTML = `<i class="fa-regular fa-message"></i> <span>${escapeHtml(c.title)}</span>`;
      item.addEventListener("click", () => selectConversation(c));
      listEl.appendChild(item);
    });

    if (!currentConversationId && convs.length > 0) {
      selectConversation(convs[0]);
    }
  } catch (err) {
    console.error("Error loading conversations:", err);
  }
}

async function selectConversation(conv) {
  currentConversationId = conv.id;
  document.getElementById("chatTitle").textContent = conv.title;
  loadConversations();

  try {
    const resp = await fetch(`/api/chats/${conv.id}/messages`);
    const msgs = await resp.json();
    const container = document.getElementById("chatMessages");
    container.innerHTML = "";

    msgs.forEach(m => {
      appendMessageCard(m.role, m.content);
    });
  } catch (err) {
    console.error("Error loading messages:", err);
  }
}

async function loadProjects() {
  try {
    const resp = await fetch("/api/projects");
    const projs = await resp.json();
    const listEl = document.getElementById("projectsList");
    listEl.innerHTML = "";

    projs.forEach(p => {
      const item = document.createElement("div");
      item.className = "nav-item";
      item.innerHTML = `<i class="fa-regular fa-folder"></i> <span>${escapeHtml(p.name)}</span>`;
      listEl.appendChild(item);
    });
  } catch (err) {
    console.error("Error loading projects:", err);
  }
}

let activeStreamBody = null;

function sendMessage() {
  const promptInput = document.getElementById("promptInput");
  const text = promptInput.value.trim();
  if (!text) return;

  if (!currentConversationId) {
    createNewConversation().then(() => doSend(text));
  } else {
    doSend(text);
  }

  promptInput.value = "";
}

function doSend(text) {
  appendMessageCard("user", text);
  activeStreamBody = appendStreamCard();

  const model = document.getElementById("modelSelect").value;
  const payload = {
    conversation_id: currentConversationId,
    message: text,
    model: model,
    mode: isCodingMode ? "coding" : "chat"
  };

  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  } else {
    setTimeout(() => {
      activeStreamBody.innerHTML = formatMarkdown(`### Local Engine Output\n\nExecuted request: ${escapeHtml(text)}`);
      activeStreamBody = null;
    }, 500);
  }
}

function appendMessageCard(role, content) {
  const container = document.getElementById("chatMessages");
  const card = document.createElement("div");
  card.className = "message-card";

  const isUser = role === "user";
  card.innerHTML = `
    <div class="message-avatar ${isUser ? "user-avatar" : "ai-avatar"}">${isUser ? "U" : "CM"}</div>
    <div class="message-body">${formatMarkdown(content)}</div>
  `;

  container.appendChild(card);
  container.scrollTop = container.scrollHeight;
}

function appendStreamCard() {
  const container = document.getElementById("chatMessages");
  const card = document.createElement("div");
  card.className = "message-card";
  card.innerHTML = `
    <div class="message-avatar ai-avatar">CM</div>
    <div class="message-body stream-content"><em>Thinking locally...</em></div>
  `;
  container.appendChild(card);
  container.scrollTop = container.scrollHeight;
  return card.querySelector(".stream-content");
}

let accumulatedStreamText = "";

function appendStreamChunk(chunk) {
  if (!activeStreamBody) return;
  if (activeStreamBody.innerHTML.includes("Thinking...")) {
    activeStreamBody.innerHTML = "";
    accumulatedStreamText = "";
  }
  accumulatedStreamText += chunk;
  activeStreamBody.innerHTML = formatMarkdown(accumulatedStreamText);
  const container = document.getElementById("chatMessages");
  container.scrollTop = container.scrollHeight;
}

function finalizeStreamMessage() {
  activeStreamBody = null;
  accumulatedStreamText = "";
}

function runEditorCode() {
  const code = document.getElementById("editorCode").value;
  if (!code.trim()) return;

  fetch("/api/code/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: code, language: "python" })
  })
  .then(res => res.json())
  .then(data => {
    appendMessageCard("assistant", `**Local Code Execution Result:**\n\`\`\`\n${data.stdout || data.stderr || "No output"}\n\`\`\``);
  });
}

function analyzeEditorCode() {
  const code = document.getElementById("editorCode").value;
  if (!code.trim()) return;

  fetch("/api/code/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: code, language: "python" })
  })
  .then(res => res.json())
  .then(data => {
    appendMessageCard("assistant", `**Local Code Analysis Report:**\n\n${data.analysis}`);
  });
}

function formatMarkdown(text) {
  if (!text) return "";
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
  formatted = formatted.replace(/\n/g, '<br>');
  return formatted;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
