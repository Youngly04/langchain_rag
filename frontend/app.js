const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const userInput = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");

function addMessage(text, sender) {
  const message = document.createElement("div");
  message.className = `message ${sender}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${sender === "user" ? "user-avatar" : "bot-avatar"}`;
  avatar.textContent = sender === "user" ? "你" : "AI";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${
    sender === "user" ? "user-bubble" : sender === "system" ? "system-bubble" : "bot-bubble"
  }`;
  bubble.textContent = text;

  if (sender === "system") {
    avatar.textContent = "!";
  }

  message.appendChild(avatar);
  message.appendChild(bubble);
  chatBox.appendChild(message);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function autoResize() {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  userInput.value = "";
  autoResize();

  addMessage("正在思考中...", "bot");
  const loadingNode = chatBox.lastChild;

  try {
    const response = await fetch("http://127.0.0.1:8010/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: text
      })
    });

    const data = await response.json();

    loadingNode.remove();

    if (!response.ok) {
      addMessage("请求失败：" + JSON.stringify(data), "system");
      return;
    }

    addMessage(data.answer  || "模型没有返回内容", "bot");
  } catch (error) {
    loadingNode.remove();
    addMessage("请求失败，请检查后端接口地址或服务状态。", "system");
    console.error(error);
  }
}

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("input", autoResize);

userInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

clearBtn.addEventListener("click", function () {
  chatBox.innerHTML = `
    <div class="message bot">
      <div class="avatar bot-avatar">AI</div>
      <div class="bubble bot-bubble">
        对话已清空，你可以重新开始提问。
      </div>
    </div>
  `;
});