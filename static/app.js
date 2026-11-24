const fileInput = document.querySelector("#file-input");
const browseButton = document.querySelector("#browse-button");
const uploadCard = document.querySelector("#upload-card");
const uploadStatus = document.querySelector("#upload-status");
const documentList = document.querySelector("#document-list");
const documentCount = document.querySelector("#document-count");
const welcome = document.querySelector("#welcome");
const messages = document.querySelector("#messages");
const questionInput = document.querySelector("#question-input");
const sendButton = document.querySelector("#send-button");
const providerLabel = document.querySelector("#provider-label");
const sessionKey = "fieldnotes-session-id";

let sessionId = localStorage.getItem(sessionKey);
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem(sessionKey, sessionId);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "Something went wrong. Please try again.");
  return data;
}

function showUploadStatus(text, isError = false) {
  uploadStatus.textContent = text;
  uploadStatus.classList.toggle("error", isError);
}

async function refreshDocuments() {
  const documents = await request("/api/documents");
  documentCount.textContent = documents.length;
  documentList.replaceChildren();
  if (!documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-docs";
    empty.innerHTML = "Your library is waiting.<br />Add a PDF to get started.";
    documentList.append(empty);
    return;
  }
  for (const item of documents) {
    const row = document.createElement("div");
    row.className = "document-item";
    const icon = document.createElement("span");
    icon.className = "pdf-icon";
    icon.textContent = "PDF";
    const name = document.createElement("span");
    name.className = "document-name";
    name.title = `${item.name} · ${item.chunks} chunks`;
    name.textContent = item.name;
    const remove = document.createElement("button");
    remove.className = "document-remove";
    remove.type = "button";
    remove.title = `Remove ${item.name}`;
    remove.setAttribute("aria-label", `Remove ${item.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", async () => {
      try {
        await request(`/api/documents/${encodeURIComponent(item.id)}`, { method: "DELETE" });
        await refreshDocuments();
      } catch (error) {
        showUploadStatus(error.message, true);
      }
    });
    row.append(icon, name, remove);
    documentList.append(row);
  }
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showUploadStatus("Please choose a PDF file.", true);
    return;
  }
  const body = new FormData();
  body.append("file", file);
  browseButton.disabled = true;
  browseButton.textContent = "Processing…";
  showUploadStatus("Reading pages, splitting text, and creating embeddings. This can take a moment.");
  try {
    const result = await request("/api/documents", { method: "POST", body });
    showUploadStatus(`Added ${result.name} · ${result.pages} pages · ${result.chunks} searchable chunks.`);
    await refreshDocuments();
  } catch (error) {
    showUploadStatus(error.message, true);
  } finally {
    browseButton.disabled = false;
    browseButton.innerHTML = 'Add a PDF <span>＋</span>';
    fileInput.value = "";
  }
}

function addMessage(role, text, sources = []) {
  const row = document.createElement("div");
  row.className = `message ${role}`;
  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "✳";
    row.append(avatar);
  }
  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;
  if (sources.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "source-list";
    for (const source of sources) {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = `${source.name} · p. ${source.page}`;
      sourceList.append(chip);
    }
    content.append(sourceList);
  }
  row.append(content);
  messages.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function sendQuestion(text = questionInput.value) {
  const question = text.trim();
  if (!question || sendButton.disabled) return;
  questionInput.value = "";
  questionInput.style.height = "auto";
  sendButton.disabled = true;
  welcome.hidden = true;
  addMessage("user", question);
  const row = document.createElement("div");
  row.className = "message assistant";
  row.innerHTML = '<div class="message-avatar">✳</div><div class="message-content typing">Searching your documents</div>';
  messages.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });
  try {
    const result = await request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: question }),
    });
    row.remove();
    addMessage("assistant", result.answer, result.sources);
  } catch (error) {
    row.remove();
    addMessage("assistant", error.message);
  } finally {
    sendButton.disabled = !questionInput.value.trim();
  }
}

browseButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));
uploadCard.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadCard.classList.add("dragover");
});
uploadCard.addEventListener("dragleave", () => uploadCard.classList.remove("dragover"));
uploadCard.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadCard.classList.remove("dragover");
  uploadFile(event.dataTransfer.files[0]);
});
sendButton.addEventListener("click", () => sendQuestion());
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 130)}px`;
  sendButton.disabled = !questionInput.value.trim();
});
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendQuestion();
  }
});
document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendQuestion(button.dataset.question));
});
document.querySelector("#clear-chat").addEventListener("click", async () => {
  await request(`/api/chat/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  messages.replaceChildren();
  welcome.hidden = false;
});

refreshDocuments().catch((error) => showUploadStatus(error.message, true));
request("/api/health")
  .then(({ provider }) => {
    providerLabel.textContent = provider === "ollama" ? "OLLAMA LOCAL" : "MISTRAL AI";
  })
  .catch(() => {
    providerLabel.textContent = "AI";
  });
