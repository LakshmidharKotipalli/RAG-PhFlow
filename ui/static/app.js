(() => {
  'use strict';

  const API_BASE = '/api/v1';
  let sessionId = null;
  let isLoading = false;

  const messagesEl = document.getElementById('messages');
  const inputEl    = document.getElementById('user-input');
  const sendBtn    = document.getElementById('send-btn');

  // ── Auto-resize textarea ────────────────────────────────────
  function resizeInput() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
  }

  inputEl.addEventListener('input', resizeInput);

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });

  sendBtn.addEventListener('click', submit);

  // ── DOM helpers ─────────────────────────────────────────────
  function appendMessage(role, content) {
    const row = document.createElement('div');
    row.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (content) bubble.textContent = content;

    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();
    return bubble;
  }

  function appendSpinner() {
    const row = document.createElement('div');
    row.className = 'message assistant';
    row.id = 'spinner-row';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    for (let i = 0; i < 3; i++) {
      spinner.appendChild(document.createElement('span'));
    }

    bubble.appendChild(spinner);
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();
    return row;
  }

  function removeSpinner() {
    const el = document.getElementById('spinner-row');
    if (el) el.remove();
  }

  function scrollToBottom() {
    const container = document.getElementById('chat-container');
    container.scrollTop = container.scrollHeight;
  }

  // ── Typewriter animation ────────────────────────────────────
  function typewrite(bubble, text, onDone) {
    let i = 0;
    bubble.textContent = '';

    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    bubble.appendChild(cursor);

    const speed = Math.max(8, Math.min(25, Math.round(6000 / text.length)));

    function tick() {
      if (i < text.length) {
        cursor.insertAdjacentText('beforebegin', text[i++]);
        scrollToBottom();
        setTimeout(tick, speed);
      } else {
        cursor.remove();
        if (onDone) onDone();
      }
    }

    tick();
  }

  // ── Source cards ────────────────────────────────────────────
  function renderSources(bubble, sources) {
    if (!sources || sources.length === 0) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'sources-wrapper';

    const label = document.createElement('div');
    label.className = 'sources-label';
    label.textContent = `Sources (${sources.length})`;
    wrapper.appendChild(label);

    sources.forEach((src) => {
      const details = document.createElement('details');
      details.className = 'source-card';

      const summary = document.createElement('summary');

      const numSpan = document.createElement('span');
      numSpan.className = 'source-section-num';
      numSpan.textContent = src.section_number || '§ —';

      const titleSpan = document.createElement('span');
      titleSpan.className = 'source-title';
      titleSpan.textContent = src.section_title || '';

      summary.appendChild(numSpan);
      summary.appendChild(titleSpan);
      details.appendChild(summary);

      if (src.full_text) {
        const body = document.createElement('div');
        body.className = 'source-body';
        const excerpt = src.full_text.length > 600
          ? src.full_text.slice(0, 600) + '\u2026'
          : src.full_text;
        body.textContent = excerpt;
        details.appendChild(body);
      }

      wrapper.appendChild(details);
    });

    bubble.appendChild(wrapper);
    scrollToBottom();
  }

  // ── API call ────────────────────────────────────────────────
  async function callChat(message) {
    const payload = {
      message,
      mode: 'qa',
      filters: {},
      stream: false,
    };

    const url = sessionId
      ? `${API_BASE}/chat/${sessionId}/message`
      : `${API_BASE}/chat`;

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        detail = err.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    return res.json();
  }

  // ── Submit handler ──────────────────────────────────────────
  async function submit() {
    if (isLoading) return;

    const text = inputEl.value.trim();
    if (!text) return;

    isLoading = true;
    sendBtn.disabled = true;
    inputEl.value = '';
    resizeInput();

    appendMessage('user', text);
    appendSpinner();

    try {
      const data = await callChat(text);
      sessionId = data.session_id;

      removeSpinner();

      const row = document.createElement('div');
      row.className = 'message assistant';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      row.appendChild(bubble);
      messagesEl.appendChild(row);
      scrollToBottom();

      typewrite(bubble, data.answer, () => {
        renderSources(bubble, data.sources);
      });

    } catch (err) {
      removeSpinner();
      const bubble = appendMessage('assistant', `Error: ${err.message}`);
      bubble.classList.add('error');
    } finally {
      isLoading = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }
})();
