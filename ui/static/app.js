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

  // ── Suggested questions ────────────────────────────────────
  document.querySelectorAll('.suggestion').forEach(btn => {
    btn.addEventListener('click', () => {
      inputEl.value = btn.dataset.query;
      resizeInput();
      // Remove all suggestions once one is clicked
      const container = document.getElementById('suggested-questions');
      if (container) container.remove();
      submit();
    });
  });

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

  // ── Structured answer rendering ─────────────────────────────
  function renderStructuredAnswer(container, text) {
    const lines = text.split('\n').filter(l => l.trim());
    if (lines.length === 0) return;

    // First line is the direct answer (Yes/No/It depends)
    const verdict = document.createElement('div');
    verdict.className = 'answer-verdict';
    verdict.textContent = lines[0];
    container.appendChild(verdict);

    // Collect numbered steps and other lines
    const steps = [];
    const other = [];
    const sopPattern = /^for facility-specific procedures/i;

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (/^\d+\.\s/.test(line)) {
        steps.push(line.replace(/^\d+\.\s*/, ''));
      } else if (sopPattern.test(line)) {
        // SOP footer handled separately below
      } else {
        other.push(line);
      }
    }

    // Render any non-step text before steps
    if (other.length > 0) {
      const extraText = document.createElement('div');
      extraText.className = 'answer-text';
      extraText.textContent = other.join(' ');
      container.appendChild(extraText);
    }

    // Render steps as a list
    if (steps.length > 0) {
      const ol = document.createElement('ol');
      ol.className = 'answer-steps';
      steps.forEach(step => {
        const li = document.createElement('li');
        li.textContent = step;
        ol.appendChild(li);
      });
      container.appendChild(ol);
    }

    // SOP footer
    const hasSop = lines.some(l => sopPattern.test(l.trim()));
    if (hasSop) {
      const footer = document.createElement('div');
      footer.className = 'answer-sop-footer';
      footer.textContent = 'For facility-specific procedures (who to notify, where to quarantine, which forms to fill), consult your site SOP.';
      container.appendChild(footer);
    }
  }

  // ── Pipeline steps ──────────────────────────────────────────
  function renderPipelineSteps(bubble, steps) {
    if (!steps || steps.length === 0) return;
    const trail = document.createElement('div');
    trail.className = 'pipeline-trail';
    steps.forEach(step => {
      const span = document.createElement('span');
      span.className = 'pipeline-step';
      span.textContent = step;
      trail.appendChild(span);
    });
    bubble.prepend(trail);
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

    sources.forEach((src, idx) => {
      const details = document.createElement('details');
      details.className = 'source-card';

      const summary = document.createElement('summary');

      const idxSpan = document.createElement('span');
      idxSpan.className = 'source-index';
      idxSpan.textContent = `[SOURCE ${idx + 1}]`;

      const numSpan = document.createElement('span');
      numSpan.className = 'source-section-num';
      numSpan.textContent = src.section_number || '§ —';

      const titleSpan = document.createElement('span');
      titleSpan.className = 'source-title';
      titleSpan.textContent = src.section_title || '';

      summary.appendChild(idxSpan);
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

  // ── API call (non-streaming, used for session init) ────────
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

  // ── API call (streaming) ──────────────────────────────────
  async function callChatStream(message, onToken, onPipelineStep) {
    const payload = {
      message,
      mode: 'qa',
      filters: {},
      stream: true,
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

    // Capture session ID from header
    const sid = res.headers.get('X-Session-Id');
    if (sid) sessionId = sid;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      // Pipeline steps come as [PIPELINE] lines
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('[PIPELINE] ')) {
          if (onPipelineStep) onPipelineStep(line.slice(11));
        } else if (line) {
          onToken(line);
        }
      }
    }
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

    // Remove suggested questions on first submit
    const suggestions = document.getElementById('suggested-questions');
    if (suggestions) suggestions.remove();

    appendMessage('user', text);

    // Show spinner while waiting for pipeline + first token
    const spinnerRow = appendSpinner();

    try {
      removeSpinner();

      const row = document.createElement('div');
      row.className = 'message assistant';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      row.appendChild(bubble);
      messagesEl.appendChild(row);

      // Pipeline steps trail (populated during streaming)
      const trail = document.createElement('div');
      trail.className = 'pipeline-trail';
      bubble.appendChild(trail);

      // Text node for streamed content
      const textNode = document.createElement('span');
      bubble.appendChild(textNode);

      const cursor = document.createElement('span');
      cursor.className = 'cursor';
      bubble.appendChild(cursor);

      scrollToBottom();

      let firstToken = true;
      let accumulated = '';
      await callChatStream(
        text,
        (token) => {
          if (firstToken) {
            removeSpinner();
            firstToken = false;
          }
          accumulated += token;
          textNode.textContent = accumulated;
          scrollToBottom();
        },
        (step) => {
          const span = document.createElement('span');
          span.className = 'pipeline-step';
          span.textContent = step;
          trail.appendChild(span);
          scrollToBottom();
        }
      );
      // Remove spinner if no tokens came (e.g. empty response)
      if (firstToken) removeSpinner();

      cursor.remove();

      // Replace raw text with structured rendering
      textNode.remove();
      renderStructuredAnswer(bubble, accumulated);
      scrollToBottom();

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
