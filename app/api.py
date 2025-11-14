# api.py — FastAPI + pretty chat UI for your salary chatbot

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent

for p in (PROJECT_ROOT, THIS_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Import your existing chatbot, unchanged
from chatbot import handle_chat, reset_context


# ---------- FastAPI app ----------

app = FastAPI(title="Salary Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = ""
    reset: bool = False


class ChatResponse(BaseModel):
    answer: str
    need_more_info: bool


# ---------- API endpoint ----------

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Simple JSON API:
    - If reset=True: clear the conversation state first.
    - Otherwise: route the message through chatbot.handle_chat.
    """
    if req.reset:
        result: Dict[str, Any] = reset_context()
    else:
        result = handle_chat(req.message)

    return ChatResponse(
        answer=result.get("answer", ""),
        need_more_info=bool(result.get("need_more_info", True)),
    )


# ---------- Frontend: pretty chat page ----------

_CHAT_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Salary Chatbot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {
      --bg: #0f172a;
      --bg-elevated: #020617;
      --accent: #4f46e5;
      --accent-soft: rgba(79, 70, 229, 0.14);
      --accent-user: #22c55e;
      --text: #e5e7eb;
      --text-muted: #9ca3af;
      --border-subtle: #1e293b;
      --radius-lg: 18px;
      --shadow-soft: 0 18px 45px rgba(15,23,42,0.8);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
        "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1e293b 0, #020617 40%, #000 100%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }

    .shell {
      width: 100%;
      max-width: 960px;
      height: 80vh;
      background: linear-gradient(145deg, #020617 0%, #020617 60%, #020617 100%);
      border-radius: 32px;
      box-shadow: var(--shadow-soft);
      border: 1px solid rgba(148, 163, 184, 0.16);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      position: relative;
    }

    .shell::before {
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at top left, rgba(79, 70, 229, 0.2), transparent 55%);
      pointer-events: none;
      opacity: 0.9;
    }

    .header {
      position: relative;
      z-index: 1;
      padding: 18px 24px;
      border-bottom: 1px solid rgba(15, 23, 42, 0.9);
      display: flex;
      align-items: center;
      justify-content: space-between;
      backdrop-filter: blur(18px);
      background: linear-gradient(to bottom, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.9));
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .avatar {
      width: 40px;
      height: 40px;
      border-radius: 999px;
      background: radial-gradient(circle at 30% 20%, #a5b4fc, #4f46e5);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 700;
      font-size: 18px;
      box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.5);
    }

    .title-block {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .title-block h1 {
      font-size: 16px;
      font-weight: 600;
      letter-spacing: 0.02em;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .pill {
      font-size: 11px;
      padding: 3px 6px;
      border-radius: 999px;
      background: rgba(34, 197, 94, 0.08);
      border: 1px solid rgba(34, 197, 94, 0.4);
      color: #bbf7d0;
    }

    .subtitle {
      font-size: 12px;
      color: var(--text-muted);
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .badge {
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.1);
      color: var(--text-muted);
      border: 1px solid rgba(148, 163, 184, 0.3);
    }

    .btn-reset {
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.4);
      background: rgba(15, 23, 42, 0.9);
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease-out;
    }

    .btn-reset:hover {
      border-color: rgba(248, 250, 252, 0.75);
      color: #e5e7eb;
      transform: translateY(-1px);
    }

    .chat-body {
      position: relative;
      z-index: 1;
      flex: 1;
      display: flex;
      padding: 16px 16px 12px;
      gap: 16px;
      overflow: hidden;
    }

    .chat-main {
      flex: 1;
      background: radial-gradient(circle at top, rgba(15, 23, 42, 0.6), rgba(15, 23, 42, 0.96));
      border-radius: 22px;
      border: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .chat-scroll {
      flex: 1;
      padding: 18px 20px 8px;
      overflow-y: auto;
      scroll-behavior: smooth;
    }

    .message-row {
      margin-bottom: 14px;
      display: flex;
      align-items: flex-end;
      gap: 8px;
    }

    .message-row.user {
      justify-content: flex-end;
    }

    .bubble {
      max-width: 70%;
      padding: 10px 12px;
      border-radius: var(--radius-lg);
      font-size: 14px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-wrap: break-word;
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.7);
    }

    .bubble.user {
      background: linear-gradient(135deg, var(--accent-user), #16a34a);
      color: #ecfdf5;
      border-bottom-right-radius: 4px;
    }

    .bubble.bot {
      background: radial-gradient(circle at top left, rgba(79, 70, 229, 0.25), rgba(15, 23, 42, 0.95));
      border: 1px solid rgba(148, 163, 184, 0.4);
      border-bottom-left-radius: 4px;
      color: #e5e7eb;
      font-size: 13px;
    }

    .bubble.bot h3,
    .bubble.bot h4 {
      margin: 4px 0 2px;
      font-size: 13px;
      font-weight: 600;
    }

    .bubble.bot ul {
      margin: 4px 0 4px 14px;
      padding-left: 0;
    }

    .bubble.bot li {
      margin-bottom: 2px;
    }

    .msg-meta {
      font-size: 11px;
      color: var(--text-muted);
      margin: 0 4px 0;
    }

    .msg-meta.bot {
      margin-left: 4px;
    }

    .msg-meta.user {
      margin-right: 4px;
    }

    .typing {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--text-muted);
    }

    .typing-dot {
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.6);
      animation: bounce 1s infinite ease-in-out;
    }

    .typing-dot:nth-child(2) {
      animation-delay: 0.15s;
    }

    .typing-dot:nth-child(3) {
      animation-delay: 0.3s;
    }

    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
      40% { transform: translateY(-4px); opacity: 1; }
    }

    .input-bar {
      padding: 10px 12px 12px;
      border-top: 1px solid var(--border-subtle);
      display: flex;
      align-items: flex-end;
      gap: 10px;
      background: rgba(15, 23, 42, 0.96);
    }

    .input-wrap {
      flex: 1;
      position: relative;
    }

    textarea {
      width: 100%;
      resize: none;
      min-height: 42px;
      max-height: 96px;
      padding: 10px 12px 10px 12px;
      border-radius: 18px;
      border: 1px solid rgba(51, 65, 85, 0.9);
      background: rgba(15, 23, 42, 0.92);
      color: #e5e7eb;
      font-size: 14px;
      line-height: 1.4;
      outline: none;
      box-shadow: inset 0 0 0 1px transparent;
    }

    textarea::placeholder {
      color: #6b7280;
    }

    textarea:focus {
      border-color: rgba(79, 70, 229, 0.8);
      box-shadow: 0 0 0 1px rgba(79, 70, 229, 0.6);
    }

    .hint {
      position: absolute;
      right: 10px;
      bottom: 6px;
      font-size: 10px;
      color: var(--text-muted);
    }

    .btn-send {
      width: 42px;
      height: 42px;
      border-radius: 999px;
      border: none;
      background: radial-gradient(circle at 30% 0, #a5b4fc, #4f46e5);
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 10px 25px rgba(79, 70, 229, 0.8);
      transition: all 0.14s ease-out;
      flex-shrink: 0;
    }

    .btn-send:disabled {
      opacity: 0.4;
      cursor: default;
      box-shadow: none;
      transform: none;
    }

    .btn-send:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 14px 30px rgba(79, 70, 229, 0.95);
    }

    .btn-send span {
      font-size: 18px;
    }

    @media (max-width: 768px) {
      .shell {
        height: 90vh;
        border-radius: 18px;
      }
      .chat-body {
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="header">
      <div class="header-left">
        <div class="avatar">S</div>
        <div class="title-block">
          <h1>
            Salary Chatbot
            <span class="pill">beta</span>
          </h1>
          <div class="subtitle">
            Describe your role and location — I'll estimate a fair base salary.
          </div>
        </div>
      </div>
      <div class="header-right">
        <div class="badge">US roles · SWE / ML / Data</div>
        <button class="btn-reset" id="btn-reset">Reset conversation</button>
      </div>
    </header>

    <main class="chat-body">
      <section class="chat-main">
        <div class="chat-scroll" id="chat-scroll">
          <!-- Messages get appended here -->
        </div>
        <div class="input-bar">
          <div class="input-wrap">
            <textarea
              id="chat-input"
              placeholder="For example: I got an offer as a Senior Software Engineer at Netflix in Los Gatos, CA."
            ></textarea>
            <div class="hint">Enter to send · Shift+Enter for a new line</div>
          </div>
          <button class="btn-send" id="btn-send">
            <span>➤</span>
          </button>
        </div>
      </section>
    </main>
  </div>

  <script>
    const API_URL = "/api/chat";

    const chatScroll = document.getElementById("chat-scroll");
    const inputEl = document.getElementById("chat-input");
    const sendBtn = document.getElementById("btn-send");
    const resetBtn = document.getElementById("btn-reset");

    let isSending = false;

    function appendMessage(role, text, meta) {
      const row = document.createElement("div");
      row.className = "message-row " + role;

      const bubble = document.createElement("div");
      bubble.className = "bubble " + role;
      
      bubble.textContent = text;

      row.appendChild(bubble);

      const metaEl = document.createElement("div");
      metaEl.className = "msg-meta " + role;
      metaEl.textContent = meta || (role === "user" ? "You" : "Bot");
      row.appendChild(metaEl);

      chatScroll.appendChild(row);
      chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function showTyping() {
      const row = document.createElement("div");
      row.className = "message-row bot";
      row.id = "typing-row";

      const bubble = document.createElement("div");
      bubble.className = "bubble bot";

      const typing = document.createElement("div");
      typing.className = "typing";
      typing.innerHTML = `
        <span>Thinking</span>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      `;

      bubble.appendChild(typing);
      row.appendChild(bubble);
      chatScroll.appendChild(row);
      chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function hideTyping() {
      const row = document.getElementById("typing-row");
      if (row) row.remove();
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text || isSending) return;

      appendMessage("user", text, "You");
      inputEl.value = "";
      autoResizeTextarea();

      isSending = true;
      sendBtn.disabled = true;
      showTyping();

      try {
        const resp = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, reset: false }),
        });

        if (!resp.ok) throw new Error("Network error");

        const data = await resp.json();
        hideTyping();
        appendMessage("bot", data.answer || "Hmm, I didn't get a response.", "Bot");
      } catch (err) {
        hideTyping();
        appendMessage(
          "bot",
          "Sorry, something went wrong talking to the API. Please try again.",
          "Bot"
        );
        console.error(err);
      } finally {
        isSending = false;
        sendBtn.disabled = false;
      }
    }

    async function resetConversation() {
      if (isSending) return;
      isSending = true;
      sendBtn.disabled = true;

      try {
        const resp = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "", reset: true }),
        });
        const data = await resp.json();

        chatScroll.innerHTML = "";
        appendMessage("bot", data.answer || "Reset complete. Tell me about your role.", "Bot");
      } catch (err) {
        appendMessage("bot", "Could not reset the conversation. Please refresh the page.", "Bot");
        console.error(err);
      } finally {
        isSending = false;
        sendBtn.disabled = false;
      }
    }

    function autoResizeTextarea() {
      inputEl.style.height = "auto";
      inputEl.style.height = Math.min(inputEl.scrollHeight, 96) + "px";
    }

    inputEl.addEventListener("input", autoResizeTextarea);

    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    sendBtn.addEventListener("click", sendMessage);
    resetBtn.addEventListener("click", resetConversation);

    // Initial greeting
    appendMessage(
      "bot",
      "Hi! I'm your salary chatbot.\\n\\n" +
        "Tell me about your role and location, for example:\\n" +
        "- `I got an offer as a Senior Software Engineer at Netflix in Los Gatos, CA`\\n" +
        "- `I'm a Staff ML Engineer at a public company in San Francisco, CA`\\n\\n" +
        "I'll estimate a reasonable base salary range and explain how I got there.",
      "Bot"
    );
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the chat page (no JSON UI, just a natural-language conversation)."""
    return HTMLResponse(_CHAT_HTML)
