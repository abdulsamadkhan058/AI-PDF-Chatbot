/* AI PDF Chatbot — production workspace shell for Chainlit 2.11 */
(() => {
  const LOGO = "/public/logo_dark.png";
  let mounted = false;
  let scheduled = false;
  let pdfState = [];
  let settingsState = { mode: "📄 PDF Only", sources: 3, memory: true };

  const norm = (v) => String(v || "").trim().toLowerCase();
  // Every rendered chat step (user message, assistant message, system bridge
  // message) carries a `data-step-type` attribute in this Chainlit build.
  const messageNodes = () => Array.from(document.querySelectorAll("[data-step-type]"));
  // Only the action-bridge buttons living inside the chat transcript should
  // ever be matched here — never the native header icons (New Chat, Readme,
  // theme toggle, ...), which can carry similar-looking labels/ids.
  const clickableNodes = () => Array.from(document.querySelectorAll('button, [role="button"]'))
    .filter((node) => !node.closest("#header"));

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  function findNativeAction(keys) {
    return clickableNodes().find((node) => {
      const text = norm([
        node.innerText, node.textContent, node.getAttribute("aria-label"),
        node.getAttribute("title"), node.getAttribute("id")
      ].join(" "));
      return keys.some((key) => text.includes(norm(key)));
    });
  }

  function triggerAction(keys) {
    const node = findNativeAction(keys);
    if (!node) return false;
    node.click();
    return true;
  }

  function injectHeader() {
    const header = document.querySelector("#header");
    if (!header) return;
    if (header.querySelector(".ai-app-brand")) return;

    header.classList.add("ai-native-header");

    // Keep the native theme toggle working, just relocate it into our bar.
    const themeToggle = header.querySelector("#theme-toggle");
    const userNav = header.querySelector("#user-nav-button");

    Array.from(header.children).forEach((child) => {
      child.classList.add("ai-native-header-hidden");
    });

    const inner = document.createElement("div");
    inner.className = "ai-app-header-inner";
    inner.innerHTML = `
      <button class="ai-mobile-menu" type="button" aria-label="Open controls">☰</button>
      <div class="ai-app-brand">
        <img src="${LOGO}" alt="AI PDF Chatbot" class="ai-brand-logo" onerror="this.style.display='none'">
        <div class="ai-brand-copy">
          <strong>AI PDF Chatbot</strong>
          <span>Smart PDF workspace</span>
        </div>
      </div>
      <div class="ai-header-status">
        <span class="ai-status-pill"><i></i> RAG</span>
        <span class="ai-status-pill">◎ Multi-language</span>
      </div>
      <div class="ai-header-actions"></div>`;
    header.appendChild(inner);
    inner.querySelector(".ai-mobile-menu").onclick = () => setSidebarOpen(true);

    const actionsSlot = inner.querySelector(".ai-header-actions");
    [themeToggle, userNav].forEach((el) => {
      if (!el) return;
      el.classList.remove("ai-native-header-hidden");
      actionsSlot.appendChild(el);
    });
  }

  function createSidebar() {
    if (mounted || document.querySelector(".ai-workspace-sidebar")) {
      mounted = true;
      return;
    }

    const overlay = document.createElement("div");
    overlay.className = "ai-sidebar-overlay";

    const sidebar = document.createElement("aside");
    sidebar.className = "ai-workspace-sidebar";
    sidebar.innerHTML = `
      <div class="ai-sidebar-head">
        <div>
          <div class="ai-sidebar-kicker">WORKSPACE</div>
          <div class="ai-sidebar-title">Controls</div>
        </div>
        <button class="ai-sidebar-close" type="button" aria-label="Close controls">×</button>
      </div>

      <button type="button" class="ai-side-btn ai-side-primary" data-ai-action="new">
        <span class="ai-side-icon">＋</span><span>New Chat</span>
      </button>

      <button type="button" class="ai-side-btn ai-control-toggle" aria-expanded="false">
        <span class="ai-side-icon">⚙</span><span>Controls</span><span class="ai-chevron">›</span>
      </button>

      <section class="ai-controls-panel" hidden>
        <div class="ai-control-group">
          <div class="ai-control-label">Answer mode</div>
          <div class="ai-segmented" role="group" aria-label="Answer mode">
            <button type="button" data-mode="pdf">PDF Only</button>
            <button type="button" data-mode="internet">PDF + Internet</button>
          </div>
        </div>

        <div class="ai-control-group">
          <div class="ai-control-row"><span class="ai-control-label">Conversation memory</span><button class="ai-switch" type="button" aria-pressed="true"><span></span></button></div>
        </div>

        <div class="ai-control-group">
          <div class="ai-control-row"><span class="ai-control-label">Max sources</span>
            <select class="ai-source-select" aria-label="Max sources">
              <option value="1">1</option><option value="2">2</option><option value="3" selected>3</option>
              <option value="4">4</option><option value="5">5</option><option value="6">6</option>
            </select>
          </div>
        </div>
      </section>

      <div class="ai-sidebar-divider"></div>
      <div class="ai-sidebar-label">WORKSPACE</div>

      <div class="ai-sidebar-actions">
        <button type="button" class="ai-side-btn" data-ai-action="clear-history"><span class="ai-side-icon">↶</span><span>Clear History</span></button>
        <button type="button" class="ai-side-btn ai-side-danger" data-ai-action="clear-pdfs"><span class="ai-side-icon">⌫</span><span>Clear PDFs</span></button>
      </div>

      <section class="ai-indexed-box">
        <div class="ai-sidebar-label ai-indexed-heading"><span>INDEXED PDFs</span><em class="ai-pdf-count">0</em></div>
        <div class="ai-indexed-list"><div class="ai-empty-index">No PDFs indexed yet</div></div>
      </section>

      <div class="ai-sidebar-footer">PDFs stay local to this running workspace.</div>`;

    document.body.appendChild(overlay);
    document.body.appendChild(sidebar);

    const close = () => setSidebarOpen(false);
    sidebar.querySelector(".ai-sidebar-close").onclick = close;
    overlay.onclick = close;

    const controlToggle = sidebar.querySelector(".ai-control-toggle");
    const controlPanel = sidebar.querySelector(".ai-controls-panel");
    controlToggle.onclick = () => {
      const open = controlToggle.getAttribute("aria-expanded") !== "true";
      controlToggle.setAttribute("aria-expanded", String(open));
      controlPanel.hidden = !open;
      controlToggle.classList.toggle("is-open", open);
    };

    sidebar.addEventListener("click", (event) => {
      const button = event.target.closest("[data-ai-action]");
      if (!button) return;
      const action = button.getAttribute("data-ai-action");
      if (action === "new") triggerAction(["New Chat"]);
      if (action === "clear-history") triggerAction(["Clear History"]);
      if (action === "clear-pdfs") triggerAction(["Clear PDFs"]);
      if (window.innerWidth <= 760) close();
    });

    sidebar.querySelectorAll("[data-mode]").forEach((button) => {
      button.onclick = () => {
        if (button.dataset.mode === "pdf") triggerAction(["PDF Only"]);
        else triggerAction(["PDF + Internet"]);
      };
    });

    sidebar.querySelector(".ai-switch").onclick = () => {
      const enabled = sidebar.querySelector(".ai-switch").getAttribute("aria-pressed") === "true";
      triggerAction([enabled ? "Memory Off" : "Memory On"]);
    };

    sidebar.querySelector(".ai-source-select").onchange = (event) => {
      triggerAction([`Sources ${event.target.value}`]);
    };

    mounted = true;
    updateSettingsUI();
  }

  function setSidebarOpen(open) {
    const sidebar = document.querySelector(".ai-workspace-sidebar");
    const overlay = document.querySelector(".ai-sidebar-overlay");
    if (!sidebar || !overlay) return;
    sidebar.classList.toggle("is-open", open);
    overlay.classList.toggle("is-open", open);
    document.body.classList.toggle("ai-sidebar-open", open);
  }

  function updateSettingsUI() {
    const sidebar = document.querySelector(".ai-workspace-sidebar");
    if (!sidebar) return;
    const mode = settingsState.mode.includes("Internet") ? "internet" : "pdf";
    sidebar.querySelectorAll("[data-mode]").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
    const sw = sidebar.querySelector(".ai-switch");
    if (sw) {
      sw.setAttribute("aria-pressed", String(settingsState.memory));
      sw.classList.toggle("active", settingsState.memory);
    }
    const select = sidebar.querySelector(".ai-source-select");
    if (select) select.value = String(settingsState.sources);
  }

  function updateSidebarPDFs() {
    const list = document.querySelector(".ai-indexed-list");
    const count = document.querySelector(".ai-pdf-count");
    if (!list || !count) return;
    count.textContent = String(pdfState.length);
    if (!pdfState.length) {
      list.innerHTML = '<div class="ai-empty-index">No PDFs indexed yet</div>';
      return;
    }
    list.innerHTML = pdfState.map((item) => `
      <div class="ai-indexed-item">
        <span class="ai-pdf-icon">PDF</span>
        <div class="ai-indexed-copy"><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.pages)} · ${escapeHtml(item.chunks)}</small></div>
      </div>`).join("");
  }

  function clearVisibleChat() {
    messageNodes().forEach((node) => {
      if (node.classList.contains("ai-persistent-welcome")) return;
      node.classList.add("ai-hidden-step");
    });
  }

  // Single pass over every rendered step: read bridge/state markers, drive
  // the sidebar from them, and hide anything that is not real conversation.
  // A MutationObserver re-runs this on every DOM change (including changes
  // caused by the user simply typing), so every marker is only ever acted
  // on ONCE (data-ai-done) — otherwise a marker still sitting in the DOM
  // (hidden, not removed) would keep re-firing its action, e.g. repeatedly
  // stomping the composer with old voice text while the user tried to edit it.
  function processSteps() {
    messageNodes().forEach((message) => {
      if (message.dataset.aiDone === "1") return;
      const text = String(message.innerText || "").trim();
      if (!text) return;

      if (/Welcome to AI PDF Chatbot/i.test(text)) {
        message.classList.add("ai-persistent-welcome");
        message.dataset.aiDone = "1";
        return;
      }

      if (text.includes("AI_ACTION_BRIDGE")) {
        // Keep it in the DOM (its action buttons are triggered
        // programmatically from the sidebar) but never show it.
        message.classList.add("ai-hidden-step");
        message.dataset.aiDone = "1";
        return;
      }

      if (text.includes("AI_SETTINGS_STATE")) {
        const parts = text.split("|").map((x) => x.trim());
        settingsState = {
          mode: parts[1] || "📄 PDF Only",
          sources: Number(parts[2]) || 3,
          memory: String(parts[3] || "true") === "true"
        };
        message.classList.add("ai-hidden-step");
        message.dataset.aiDone = "1";
        updateSettingsUI();
        return;
      }

      if (text.includes("AI_PDF_INDEX_STATE")) {
        const items = text.split(/\n+/).map((line) => line.trim()).filter((line) => line.includes("AI_PDF_INDEX_ITEM"));
        pdfState = items.map((line) => {
          const parts = line.split("|").map((x) => x.trim());
          return { name: parts[1] || "PDF", pages: parts[2] || "0 pages", chunks: parts[3] || "0 chunks" };
        });
        message.classList.add("ai-hidden-step");
        message.dataset.aiDone = "1";
        updateSidebarPDFs();
        return;
      }

      if (text.includes("AI_UI_CLEAR_HISTORY") || text.includes("AI_UI_NEW_CHAT")) {
        message.classList.add("ai-hidden-step");
        message.dataset.aiDone = "1";
        clearVisibleChat();
        return;
      }

      if (text.includes("AI_UI_CLEAR_PDFS")) {
        message.classList.add("ai-hidden-step");
        message.dataset.aiDone = "1";
        pdfState = [];
        updateSidebarPDFs();
        clearVisibleChat();
        return;
      }

      if (text.includes("AI_VOICE_TRANSCRIPT")) {
        message.classList.add("ai-hidden-step");
        message.dataset.aiDone = "1";
        const spoken = text.split("|").slice(1).join("|").trim();
        if (spoken) setComposerText(spoken);
        return;
      }

      if (/PDF\(s\) ready|Indexed PDFs/i.test(text)) {
        message.classList.add("ai-hidden-step");
        message.dataset.aiDone = "1";
      }
    });
  }

  // Puts transcribed speech into the actual message box (like Google/WhatsApp
  // voice-to-text) so the user can see it, edit it, and press Send themselves.
  // Only ever called once per recording (see data-ai-done above) — that,
  // not the injection technique itself, was the real cause of the "stuck"
  // input, since without it this ran again on every keystroke.
  function setComposerText(text) {
    const input = document.querySelector("#chat-input");
    if (!input) return;
    input.focus();

    if (input.tagName === "TEXTAREA" || input.tagName === "INPUT") {
      const proto = input.tagName === "TEXTAREA"
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
      setter.call(input, text);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      // Rich-text (contenteditable) composer: go through the real
      // browser text-input pipeline so the editor's own state updates,
      // instead of a plain textContent assignment it would ignore.
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(input);
      selection.removeAllRanges();
      selection.addRange(range);
      const inserted = document.execCommand && document.execCommand("insertText", false, text);
      if (!inserted) {
        input.textContent = text;
        input.dispatchEvent(new InputEvent("input", { bubbles: true, data: text, inputType: "insertText" }));
      }
    }

    if (typeof input.setSelectionRange === "function") {
      try { input.setSelectionRange(text.length, text.length); } catch (e) { /* no-op */ }
    }
  }

  function fixComposer() {
    const input = document.querySelector("#chat-input");
    if (input) {
      input.style.boxSizing = "border-box";
      input.style.textIndent = "0";
      input.style.transform = "none";
      input.style.paddingLeft = "2px";
    }
  }

  function refresh() {
    if (scheduled) return;
    scheduled = true;
    setTimeout(() => {
      scheduled = false;
      injectHeader();
      createSidebar();
      processSteps();
      fixComposer();
    }, 30);
  }

  function start() {
    refresh();
    const observer = new MutationObserver(refresh);
    observer.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("resize", refresh, { passive: true });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();
