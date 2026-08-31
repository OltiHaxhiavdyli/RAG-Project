// Same-origin: served by the same FastAPI app that exposes /health, /query,
// /ingest, /session/{id}, so no base URL or CORS config is needed. Plain
// fetch() also has no default timeout (unlike Python's `requests`), so
// unlike the old Streamlit client there's nothing to configure to survive
// a 30-90s query.
//
// /query/stream sends real progress as newline-delimited JSON while
// ChatSession.ask() actually moves through each stage (see pipeline.py's
// on_stage callback) — this is not a client-side timer guessing at canned
// labels; each transition below fires only when the backend really reaches
// it.

const STAGE_LABELS = {
  initializing: "Starting up (first question in this session — building the retrieval index)",
  routing: "Routing the question",
  querying_database: "Querying the database (text-to-SQL)",
  retrieving: "Retrieving documents",
  generating: "Generating an answer",
  verifying: "Verifying the answer",
  regenerating: "Regenerating (previous answer had unsupported claims)",
  rewriting_question: "Rewriting the question for a better retrieval pass",
};

const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const cancelBtn = document.getElementById("cancel-btn");
const statusEl = document.getElementById("status");
const reingestBtn = document.getElementById("reingest-btn");
const newChatBtn = document.getElementById("new-chat-btn");

let sessionId = localStorage.getItem("rag_session_id");
let messages = JSON.parse(localStorage.getItem("rag_messages") || "[]");
let controller = null;

function saveState() {
  if (sessionId) localStorage.setItem("rag_session_id", sessionId);
  else localStorage.removeItem("rag_session_id");
  localStorage.setItem("rag_messages", JSON.stringify(messages));
}

function renderMessage(m) {
  const wrap = document.createElement("div");
  wrap.className = `message ${m.role}${m.error ? " error" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = m.content;
  wrap.appendChild(bubble);

  if (m.role === "assistant" && (m.route || (m.sources && m.sources.length))) {
    const details = document.createElement("details");
    details.className = "details";
    const summary = document.createElement("summary");
    summary.textContent = "Details";
    details.appendChild(summary);

    if (m.route) {
      const routeEl = document.createElement("div");
      routeEl.textContent = `Route: ${m.route}`;
      details.appendChild(routeEl);
    }
    if (m.sources && m.sources.length) {
      const list = document.createElement("ul");
      for (const s of m.sources) {
        const li = document.createElement("li");
        li.textContent = s;
        list.appendChild(li);
      }
      details.appendChild(list);
    }
    wrap.appendChild(details);
  }
  return wrap;
}

function renderMessages() {
  messagesEl.innerHTML = "";
  for (const m of messages) messagesEl.appendChild(renderMessage(m));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addPendingBubble() {
  const wrap = document.createElement("div");
  wrap.className = "message assistant pending";
  wrap.id = "pending-message";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error();
    const data = await res.json();
    statusEl.textContent = `API up — ${data.indexed_chunks} chunks indexed`;
    statusEl.className = "status ok";
  } catch {
    statusEl.textContent = "Can't reach the API.";
    statusEl.className = "status error";
  }
}

async function sendQuestion(question) {
  messages.push({ role: "user", content: question });
  renderMessages();
  saveState();

  const bubble = addPendingBubble();
  let elapsed = 0;
  // Honest neutral default: no real stage event has arrived yet, so this
  // must not presume any specific stage (e.g. "routing") is happening.
  let stageLabel = "Sending";
  const renderBubble = () => {
    bubble.textContent = `${stageLabel}... (${elapsed}s)`;
  };
  renderBubble();
  const timer = setInterval(() => {
    elapsed += 1;
    renderBubble();
  }, 1000);

  controller = new AbortController();
  sendBtn.hidden = true;
  cancelBtn.hidden = false;
  input.disabled = true;

  let reply;
  try {
    const res = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `Request failed (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalData = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newlineIndex;
      while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newlineIndex).trim();
        buffer = buffer.slice(newlineIndex + 1);
        if (!line) continue;

        const event = JSON.parse(line);
        if (event.error) {
          throw new Error(event.error);
        } else if (event.stage) {
          stageLabel = STAGE_LABELS[event.stage] || event.stage;
          renderBubble();
        } else {
          finalData = event; // the final result: {answer, sources, route, session_id}
        }
      }
    }

    if (!finalData) throw new Error("Stream ended without a result.");
    sessionId = finalData.session_id;
    reply = {
      role: "assistant",
      content: finalData.answer,
      route: finalData.route,
      sources: finalData.sources,
    };
  } catch (e) {
    if (e.name === "AbortError") {
      // Only stops the browser from waiting — the backend has no
      // cancellation hook, so the query keeps running server-side
      // regardless. Said plainly here rather than implying otherwise.
      reply = {
        role: "assistant",
        error: true,
        content: "Cancelled waiting. The server may still be finishing this query in the background.",
      };
    } else {
      reply = { role: "assistant", error: true, content: `Error: ${e.message}` };
    }
  } finally {
    clearInterval(timer);
    document.getElementById("pending-message")?.remove();
    sendBtn.hidden = false;
    cancelBtn.hidden = true;
    input.disabled = false;
    controller = null;
  }

  messages.push(reply);
  renderMessages();
  saveState();
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || controller) return;
  input.value = "";
  sendQuestion(question);
});

cancelBtn.addEventListener("click", () => {
  if (controller) controller.abort();
});

reingestBtn.addEventListener("click", async () => {
  reingestBtn.disabled = true;
  const originalLabel = reingestBtn.textContent;
  reingestBtn.textContent = "Ingesting...";
  try {
    const res = await fetch("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Ingest failed");
    alert(`Added ${data.chunks_added} chunks (${data.total_chunks} total).`);
    checkHealth();
  } catch (e) {
    alert(`Ingest failed: ${e.message}`);
  } finally {
    reingestBtn.disabled = false;
    reingestBtn.textContent = originalLabel;
  }
});

newChatBtn.addEventListener("click", async () => {
  if (sessionId) {
    try {
      await fetch(`/session/${sessionId}`, { method: "DELETE" });
    } catch {
      // best-effort cleanup of the server-side in-memory session; a stale
      // one isn't worth blocking on
    }
  }
  sessionId = null;
  messages = [];
  saveState();
  renderMessages();
});

renderMessages();
checkHealth();
