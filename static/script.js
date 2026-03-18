/* ============================================
   5GNU 智能 Agent - 前端交互逻辑
   ============================================ */

const API_URL = "/chat";
const WELCOME_MSG =
  "您好！我是 **5GNU（5G新媒体）** 的专属智能 Agent 🚀\n\n" +
  "关于以下话题，我都可以为您详细解答：\n" +
  "- 🎓 **无人机 STEAM 教育** 与 AOPA 考证流程\n" +
  "- 🏟️ **天地足球赛事**（无人机足球 + 机器人足球）\n" +
  "- 🏫 **LAEC 低空经济学校中心** 项目介绍\n" +
  "- 📡 **御空 5G 智能机场系统** 技术方案\n" +
  "- 💻 **编程辅导**、科技常识等通用问题\n\n" +
  "请问今天有什么我可以帮您的？✨";

const chatContainer = document.getElementById("chatContainer");
const userInput     = document.getElementById("userInput");
const sendBtn       = document.getElementById("sendBtn");
const clearBtn      = document.getElementById("clearBtn");

let messageHistory = [];
let isStreaming    = false;

// marked.js 配置
marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: function(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try { return hljs.highlight(code, { language: lang }).value; } catch(e) {}
    }
    return hljs.highlightAuto(code).value;
  }
});

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function autoResize() {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
}

function appendMessage(role, content = "") {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}-avatar`;
  avatar.textContent = role === "bot" ? "🤖" : "👤";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}-bubble`;

  if (role === "user") {
    bubble.textContent = content;
  } else if (content) {
    bubble.innerHTML = marked.parse(content);
    hljs.highlightAll();
  } else {
    bubble.innerHTML = `
      <div class="thinking">
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
        <span style="margin-left:6px">思考中...</span>
      </div>`;
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatContainer.appendChild(row);
  scrollToBottom();
  return bubble;
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isStreaming) return;

  isStreaming = true;
  sendBtn.disabled = true;

  appendMessage("user", text);
  userInput.value = "";
  userInput.style.height = "auto";

  messageHistory.push({ role: "user", content: text });

  const botBubble = appendMessage("bot");

  let fullText = "";
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: messageHistory }),
    });

    if (!response.ok) throw new Error(`HTTP 错误：${response.status}`);

    const reader  = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    botBubble.innerHTML = "";
    botBubble.classList.add("typing-cursor");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const dataStr = line.replace(/^data: /, "").trim();
        if (dataStr === "[DONE]") break;

        try {
          const json = JSON.parse(dataStr);
          if (json.content) {
            fullText += json.content;
            botBubble.innerHTML = marked.parse(fullText);
            scrollToBottom();
          }
          if (json.error) {
            botBubble.classList.remove("typing-cursor");
            botBubble.innerHTML = `<span style="color:#f85149">⚠️ 服务异常：${json.error}</span>`;
            break;
          }
        } catch (e) {}
      }
    }

    botBubble.classList.remove("typing-cursor");
    if (fullText) {
      botBubble.innerHTML = marked.parse(fullText);
      botBubble.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));
    }

    if (fullText) {
      messageHistory.push({ role: "assistant", content: fullText });
    }

  } catch (err) {
    botBubble.classList.remove("typing-cursor");
    botBubble.innerHTML = `<span style="color:#f85149">⚠️ 网络错误，请稍后重试。</span>`;
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    scrollToBottom();
    userInput.focus();
  }
}

/* ────────────────────────────────────────────
   清空对话逻辑
   ──────────────────────────────────────────── */
clearBtn.addEventListener("click", () => {
  if (isStreaming) return; // 如果正在回答，不让清空
  
  // 1. 清空 DOM
  chatContainer.innerHTML = "";
  
  // 2. 重置历史记录数组
  messageHistory = [];
  
  // 3. 重新插入欢迎语
  appendMessage("bot", WELCOME_MSG);
  
  // 4. 重置输入框状态
  userInput.value = "";
  autoResize();
  userInput.focus();
});

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

userInput.addEventListener("input", autoResize);

window.addEventListener("DOMContentLoaded", () => {
  appendMessage("bot", WELCOME_MSG);
  userInput.focus();
});
