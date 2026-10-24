/**
 * ARIA // STUDIO — Minimalist ChatGPT UI/UX Client Script
 *
 * Architecture:
 * - Multi-session Local Storage persistence
 * - Server-Sent Events (SSE) word streaming with fallback to synchronous POST
 * - Real-time Markdown parser with offline code-block preservation
 * - Dynamic Code block copy buttons
 * - Inspect / Debug confidence HUD
 * - Model retraining trigger and live diagnostics modal
 */

(function () {
  "use strict";

  // ===========================================================================
  // DOM ELEMENT REFERENCES
  // ===========================================================================
  const chatViewport = document.getElementById("chatViewport");
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const chatHistoryList = document.getElementById("chatHistoryList");
  const stopWrapper = document.getElementById("stopWrapper");
  const stopBtn = document.getElementById("stopBtn");
  const sidebar = document.getElementById("sidebar");
  const openSidebarBtn = document.getElementById("openSidebarBtn");
  const closeSidebarBtn = document.getElementById("closeSidebarBtn");
  const debugToggleBtn = document.getElementById("debugToggleBtn");
  const retrainTopBtn = document.getElementById("retrainTopBtn");
  const settingsModal = document.getElementById("settingsModal");
  const openSettingsBtn = document.getElementById("openSettingsBtn");
  const closeSettingsModalBtn = document.getElementById("closeSettingsModalBtn");
  const modalRetrainBtn = document.getElementById("modalRetrainBtn");
  const retrainStatusMsg = document.getElementById("retrainStatusMsg");
  const modalIntentsCount = document.getElementById("modalIntentsCount");
  const toastContainer = document.getElementById("toastContainer");

  // ===========================================================================
  // APPLICATION STATE & STORAGE KEYS
  // ===========================================================================
  let sessions = [];
  let currentSessionId = null;
  let isGenerating = false;
  let abortController = null;
  let debugMode = false;

  const STORAGE_KEY = "chatgpt_py_sessions";
  const DEBUG_KEY = "chatgpt_py_debug";

  // ===========================================================================
  // MARKDOWN & TEXT RENDERING UTILITIES
  // ===========================================================================

  function parseMarkdown(text) {
    if (window.marked && typeof window.marked.parse === "function") {
      try {
        return window.marked.parse(text);
      } catch (e) {
        console.warn("Marked.js error, using fallback:", e);
      }
    }

    // Robust offline fallback formatter
    let escaped = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Extract code blocks first to protect them from line break conversions
    const codeBlocks = [];
    escaped = escaped.replace(/```([a-z0-9_-]*)\n?([\s\S]*?)```/gi, (match, lang, code) => {
      const marker = `___CODE_BLOCK_${codeBlocks.length}___`;
      codeBlocks.push(`<pre><code class="language-${lang || 'plaintext'}">${code}</code></pre>`);
      return marker;
    });

    // Inline code
    escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Bold
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Italic
    escaped = escaped.replace(/\*(.*?)\*/g, "<em>$1</em>");
    // Line breaks
    escaped = escaped.replace(/\n/g, "<br>");

    // Restore protected code blocks
    codeBlocks.forEach((block, idx) => {
      escaped = escaped.replace(`___CODE_BLOCK_${idx}___`, block);
    });

    return `<p>${escaped}</p>`;
  }

  function showToast(message, duration = 3000) {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 200);
    }, duration);
  }

  // ===========================================================================
  // LIFECYCLE & SETTINGS
  // ===========================================================================

  function init() {
    loadSettings();
    loadSessions();
    setupEventListeners();
    fetchModelStatus();
  }

  function loadSettings() {
    debugMode = localStorage.getItem(DEBUG_KEY) === "true";
    updateDebugUI();
  }

  function updateDebugUI() {
    if (!debugToggleBtn) return;
    if (debugMode) {
      debugToggleBtn.classList.add("active");
    } else {
      debugToggleBtn.classList.remove("active");
    }
  }

  // ===========================================================================
  // SESSION MANAGEMENT
  // ===========================================================================

  function loadSessions() {
    try {
      const data = localStorage.getItem(STORAGE_KEY);
      sessions = data ? JSON.parse(data) : [];
    } catch (e) {
      sessions = [];
    }

    if (sessions.length === 0) {
      createNewSession();
    } else {
      currentSessionId = sessions[0].id;
      renderHistoryList();
      renderCurrentSession();
    }
  }

  function saveSessions() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch (e) {
      console.error("Failed to save sessions:", e);
    }
  }

  function getActiveSession() {
    return sessions.find(s => s.id === currentSessionId);
  }

  function createNewSession() {
    const newSession = {
      id: "sess_" + Date.now(),
      title: "New chat",
      createdAt: new Date().toISOString(),
      messages: []
    };
    sessions.unshift(newSession);
    currentSessionId = newSession.id;
    saveSessions();
    renderHistoryList();
    renderCurrentSession();
    if (chatInput) chatInput.focus();
  }

  function renderHistoryList() {
    if (!chatHistoryList) return;
    chatHistoryList.innerHTML = "";
    sessions.forEach(session => {
      const item = document.createElement("div");
      item.className = "history-item" + (session.id === currentSessionId ? " active" : "");
      item.dataset.id = session.id;

      const titleSpan = document.createElement("span");
      titleSpan.className = "history-item-title";
      titleSpan.textContent = session.title || "Untitled chat";

      const actionsDiv = document.createElement("div");
      actionsDiv.className = "history-item-actions";

      // Delete Button
      const delBtn = document.createElement("button");
      delBtn.className = "history-action-btn";
      delBtn.title = "Delete chat";
      delBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none">
          <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
        </svg>
      `;
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(session.id);
      });

      actionsDiv.appendChild(delBtn);
      item.appendChild(titleSpan);
      item.appendChild(actionsDiv);

      item.addEventListener("click", () => {
        if (session.id !== currentSessionId) {
          currentSessionId = session.id;
          renderHistoryList();
          renderCurrentSession();
        }
      });

      chatHistoryList.appendChild(item);
    });
  }

  function deleteSession(id) {
    sessions = sessions.filter(s => s.id !== id);
    if (sessions.length === 0) {
      createNewSession();
    } else {
      if (currentSessionId === id) {
        currentSessionId = sessions[0].id;
      }
      saveSessions();
      renderHistoryList();
      renderCurrentSession();
    }
  }

  function renderCurrentSession() {
    const session = getActiveSession();
    if (!session || session.messages.length === 0) {
      if (welcomeScreen) welcomeScreen.style.display = "flex";
      if (messagesContainer) messagesContainer.innerHTML = "";
    } else {
      if (welcomeScreen) welcomeScreen.style.display = "none";
      if (messagesContainer) {
        messagesContainer.innerHTML = "";
        session.messages.forEach(msg => {
          if (msg.role === "user") {
            appendUserMessageDOM(msg.content);
          } else {
            appendAssistantMessageDOM(msg.content, msg.metadata, false);
          }
        });
      }
      scrollToBottom();
    }
  }

  // ===========================================================================
  // MESSAGE RENDERING & CODE HIGHLIGHTING
  // ===========================================================================

  function appendUserMessageDOM(text) {
    if (!messagesContainer) return;
    const row = document.createElement("div");
    row.className = "message-row-user";

    const bubble = document.createElement("div");
    bubble.className = "user-bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    messagesContainer.appendChild(row);
    if (welcomeScreen) welcomeScreen.style.display = "none";
    scrollToBottom();
  }

  function appendAssistantMessageDOM(rawContent, metadata, isLiveStreaming = false) {
    if (!messagesContainer) return {};
    const row = document.createElement("div");
    row.className = "message-row-assistant";

    const avatar = document.createElement("div");
    avatar.className = "assistant-avatar";
    avatar.innerHTML = `<span>A</span>`;

    const wrapper = document.createElement("div");
    wrapper.className = "assistant-content-wrapper";

    const body = document.createElement("div");
    body.className = "assistant-body";

    wrapper.appendChild(body);

    // Actions bar (Copy, Metadata pills)
    const actions = document.createElement("div");
    actions.className = "message-actions";

    const copyBtn = document.createElement("button");
    copyBtn.className = "action-icon-btn";
    copyBtn.title = "Copy message";
    copyBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
      </svg>
      <span>Copy</span>
    `;
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(rawContent).then(() => {
        copyBtn.innerHTML = `
          <svg viewBox="0 0 24 24" width="14" height="14" stroke="#ffffff" stroke-width="2.5" fill="none">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
          <span style="color: #ffffff;">Copied</span>
        `;
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
            <span>Copy</span>
          `;
        }, 2000);
      });
    });
    actions.appendChild(copyBtn);

    // Append metadata debug tag
    if (metadata) {
      const confPill = document.createElement("span");
      const confPercent = Math.round((metadata.confidence || 0) * 100);
      confPill.className = "debug-tag";
      const engineName = metadata.engine ? ` [${metadata.engine}]` : "";
      confPill.textContent = `${metadata.intent || "unknown"}${engineName} (${confPercent}%)`;
      confPill.style.display = debugMode ? "inline-block" : "none";
      confPill.dataset.debugTag = "true";
      actions.appendChild(confPill);
    }

    wrapper.appendChild(actions);
    row.appendChild(avatar);
    row.appendChild(wrapper);
    messagesContainer.appendChild(row);

    if (!isLiveStreaming) {
      renderFormattedContent(body, rawContent);
    }

    return { row, body, actions, wrapper };
  }

  function renderFormattedContent(container, markdownText) {
    container.innerHTML = parseMarkdown(markdownText);

    // Format code blocks with ChatGPT-style top bar and Copy button
    const preBlocks = container.querySelectorAll("pre");
    preBlocks.forEach(pre => {
      const codeEl = pre.querySelector("code");
      const codeText = codeEl ? codeEl.innerText : pre.innerText;
      let lang = "Code";
      if (codeEl && codeEl.className) {
        const match = codeEl.className.match(/language-([a-z0-9_-]+)/i);
        if (match) lang = match[1];
      }

      const wrapper = document.createElement("div");
      wrapper.className = "code-block-wrapper";

      const header = document.createElement("div");
      header.className = "code-header";
      header.innerHTML = `
        <span>${lang}</span>
        <button class="copy-code-btn">
          <svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" stroke-width="2" fill="none">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
          </svg>
          <span>Copy code</span>
        </button>
      `;

      const copyBtn = header.querySelector(".copy-code-btn");
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(codeText).then(() => {
          copyBtn.innerHTML = `<span>Copied!</span>`;
          setTimeout(() => {
            copyBtn.innerHTML = `
              <svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" stroke-width="2" fill="none">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
              </svg>
              <span>Copy code</span>
            `;
          }, 2000);
        });
      });

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
    });
  }

  function scrollToBottom(force = false) {
    if (!chatViewport) return;
    const isNearBottom = chatViewport.scrollHeight - chatViewport.scrollTop - chatViewport.clientHeight < 140;
    if (force || isNearBottom) {
      chatViewport.scrollTop = chatViewport.scrollHeight;
    }
  }

  // ===========================================================================
  // QUERY DISPATCH & STREAMING (SSE)
  // ===========================================================================

  async function sendMessage(text) {
    if (!text || !text.trim() || isGenerating) return;

    const userQuery = text.trim();
    if (chatInput) chatInput.value = "";
    adjustTextareaHeight();
    updateSendBtnState();

    const session = getActiveSession();
    if (!session) return;

    // Auto-name chat if first message
    if (session.messages.length === 0) {
      session.title = userQuery.length > 28 ? userQuery.slice(0, 28) + "..." : userQuery;
      renderHistoryList();
    }

    // Save user message to session
    session.messages.push({
      role: "user",
      content: userQuery,
      timestamp: new Date().toISOString()
    });
    saveSessions();

    appendUserMessageDOM(userQuery);

    // Setup Assistant Streaming UI
    isGenerating = true;
    if (stopWrapper) stopWrapper.style.display = "flex";
    abortController = new AbortController();

    const { body, actions } = appendAssistantMessageDOM("", null, true);
    let streamedText = "";
    let metadata = null;

    if (body) body.innerHTML = '<span class="streaming-cursor"></span>';
    scrollToBottom();

    const engineSelect = document.getElementById("engineSelector");
    const selectedEngine = engineSelect ? engineSelect.value : "hybrid";

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userQuery, engine: selectedEngine }),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === "meta") {
                metadata = data;
              } else if (data.type === "chunk") {
                streamedText += data.content;
                if (body) {
                  renderFormattedContent(body, streamedText);
                  body.innerHTML += '<span class="streaming-cursor"></span>';
                }
                scrollToBottom();
              } else if (data.type === "done") {
                break;
              }
            } catch (err) {
              console.error("Parse line error:", err);
            }
          }
        }
      }

    } catch (err) {
      if (err.name === "AbortError") {
        console.log("Stream generation stopped by user.");
      } else {
        console.error("Stream failed, attempting fallback:", err);
        try {
          const fallbackRes = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userQuery, engine: selectedEngine })
          });
          const fallbackData = await fallbackRes.json();
          streamedText = fallbackData.reply || "No reply received.";
          metadata = fallbackData;
        } catch (e2) {
          streamedText = "Sorry, an error occurred while processing your message.";
        }
      }
    } finally {
      isGenerating = false;
      if (stopWrapper) stopWrapper.style.display = "none";
      abortController = null;

      // Finalize formatted markdown and clean cursor
      if (body) renderFormattedContent(body, streamedText);

      // Append metadata debug tag
      if (metadata && actions) {
        const confPill = document.createElement("span");
        const confPercent = Math.round((metadata.confidence || 0) * 100);
        confPill.className = "debug-tag";
        const engineName = metadata.engine ? ` [${metadata.engine}]` : "";
        confPill.textContent = `${metadata.intent || "unknown"}${engineName} (${confPercent}%)`;
        confPill.style.display = debugMode ? "inline-block" : "none";
        confPill.dataset.debugTag = "true";
        actions.appendChild(confPill);
      }

      // Save assistant message to session
      session.messages.push({
        role: "assistant",
        content: streamedText,
        metadata: metadata,
        timestamp: new Date().toISOString()
      });
      saveSessions();
      scrollToBottom();
    }
  }

  // ===========================================================================
  // MODEL DIAGNOSTICS & RETRAINING
  // ===========================================================================

  async function fetchModelStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (modalIntentsCount) {
        modalIntentsCount.textContent = `${data.intents_count} tags (${data.available_intents ? data.available_intents.join(", ") : ""})`;
      }
      const ragChunksEl = document.getElementById("modalAriaRagChunks");
      if (ragChunksEl && data.aria_rag_chunks !== undefined) {
        ragChunksEl.textContent = `${data.aria_rag_chunks} Indexed Chunks (BM25)`;
      }
    } catch (err) {
      console.warn("Unable to fetch status:", err);
    }
  }

  async function triggerRetrain() {
    if (retrainStatusMsg) {
      retrainStatusMsg.style.color = "var(--text-secondary)";
      retrainStatusMsg.textContent = "Compiling weights on local dataset...";
    }
    showToast("Compiling model weights...");

    try {
      const res = await fetch("/api/retrain", { method: "POST" });
      const data = await res.json();
      if (data.success) {
        if (retrainStatusMsg) {
          retrainStatusMsg.style.color = "#34d399";
          retrainStatusMsg.textContent = `Weights compiled successfully (${data.accuracy}% accuracy)`;
        }
        showToast(`Compilation complete (${data.accuracy}% accuracy)`);
        fetchModelStatus();
      } else {
        throw new Error(data.error || "Retrain failed");
      }
    } catch (err) {
      if (retrainStatusMsg) {
        retrainStatusMsg.style.color = "#f87171";
        retrainStatusMsg.textContent = `Error: ${err.message}`;
      }
      showToast(`Error: ${err.message}`);
    }
  }

  // ===========================================================================
  // EVENT LISTENERS & UI CONTROLS
  // ===========================================================================

  function setupEventListeners() {
    if (chatInput) {
      chatInput.addEventListener("input", () => {
        adjustTextareaHeight();
        updateSendBtnState();
      });

      chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendMessage(chatInput.value);
        }
      });
    }

    if (sendBtn) {
      sendBtn.addEventListener("click", () => {
        if (chatInput) sendMessage(chatInput.value);
      });
    }

    if (newChatBtn) {
      newChatBtn.addEventListener("click", () => {
        createNewSession();
      });
    }

    if (stopBtn) {
      stopBtn.addEventListener("click", () => {
        if (abortController) {
          abortController.abort();
        }
      });
    }

    document.querySelectorAll(".suggestion-card").forEach(card => {
      card.addEventListener("click", () => {
        const prompt = card.dataset.prompt;
        if (prompt) sendMessage(prompt);
      });
    });

    if (openSidebarBtn && sidebar) {
      openSidebarBtn.addEventListener("click", () => {
        sidebar.classList.toggle("collapsed");
        sidebar.classList.toggle("open");
      });
    }

    if (closeSidebarBtn && sidebar) {
      closeSidebarBtn.addEventListener("click", () => {
        sidebar.classList.add("collapsed");
        sidebar.classList.remove("open");
      });
    }

    if (debugToggleBtn) {
      debugToggleBtn.addEventListener("click", () => {
        debugMode = !debugMode;
        localStorage.setItem(DEBUG_KEY, String(debugMode));
        updateDebugUI();

        document.querySelectorAll("[data-debug-tag='true']").forEach(tag => {
          tag.style.display = debugMode ? "inline-block" : "none";
        });

        showToast(`Inspect Mode: ${debugMode ? "Enabled" : "Disabled"}`);
      });
    }

    if (retrainTopBtn) {
      retrainTopBtn.addEventListener("click", () => {
        triggerRetrain();
      });
    }

    if (openSettingsBtn && settingsModal) {
      openSettingsBtn.addEventListener("click", () => {
        fetchModelStatus();
        if (retrainStatusMsg) retrainStatusMsg.textContent = "";
        settingsModal.style.display = "flex";
      });
    }

    if (closeSettingsModalBtn && settingsModal) {
      closeSettingsModalBtn.addEventListener("click", () => {
        settingsModal.style.display = "none";
      });
    }

    if (modalRetrainBtn) {
      modalRetrainBtn.addEventListener("click", () => {
        triggerRetrain();
      });
    }

    if (settingsModal) {
      window.addEventListener("click", (e) => {
        if (e.target === settingsModal) {
          settingsModal.style.display = "none";
        }
      });
    }
  }

  function adjustTextareaHeight() {
    if (!chatInput) return;
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + "px";
  }

  function updateSendBtnState() {
    if (!chatInput || !sendBtn) return;
    const hasText = chatInput.value.trim().length > 0;
    sendBtn.disabled = !hasText;
    if (hasText) {
      sendBtn.classList.add("active");
    } else {
      sendBtn.classList.remove("active");
    }
  }

  // Initialize on script load
  init();
})();
