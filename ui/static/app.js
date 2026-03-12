(() => {
  'use strict';

  const API = '/api/v1';
  let sessionId = null;
  let busy = false;

  const $messages = document.getElementById('messages');
  const $input    = document.getElementById('user-input');
  const $send     = document.getElementById('send-btn');

  // ── Textarea auto-resize ───────────────────────────────────
  function resize() {
    $input.style.height = 'auto';
    $input.style.height = Math.min($input.scrollHeight, 160) + 'px';
  }
  $input.addEventListener('input', resize);
  $input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $send.addEventListener('click', send);

  // ── Suggestion chips ───────────────────────────────────────
  document.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      $input.value = btn.dataset.q;
      resize();
      send();
    });
  });

  // ── Scroll helper ──────────────────────────────────────────
  function scroll() {
    const c = document.getElementById('chat-container');
    c.scrollTop = c.scrollHeight;
  }

  // ── DOM: user bubble ───────────────────────────────────────
  function addUser(text) {
    const row = el('div', 'message user');
    const b = el('div', 'bubble');
    b.textContent = text;
    row.appendChild(b);
    $messages.appendChild(row);
    scroll();
  }

  // ── DOM: spinner ───────────────────────────────────────────
  function addSpinner() {
    const row = el('div', 'message assistant', 'spinner-row');
    const b = el('div', 'bubble');
    const s = el('div', 'spinner');
    for (let i = 0; i < 3; i++) s.appendChild(document.createElement('span'));
    b.appendChild(s);
    row.appendChild(b);
    $messages.appendChild(row);
    scroll();
  }
  function removeSpinner() {
    const s = document.getElementById('spinner-row');
    if (s) s.remove();
  }

  // ── DOM: tiny helper ───────────────────────────────────────
  function el(tag, cls, id) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (id) e.id = id;
    return e;
  }

  // ── Streaming API call ─────────────────────────────────────
  async function stream(message, onPipeline, onToken, onSources) {
    const url = sessionId ? `${API}/chat/${sessionId}/message` : `${API}/chat`;
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message, mode: 'qa', filters: {}, stream: true }),
    });

    if (!res.ok) {
      let d = `HTTP ${res.status}`;
      try { d = (await res.json()).detail || d; } catch {}
      throw new Error(d);
    }

    const sid = res.headers.get('X-Session-Id');
    if (sid) sessionId = sid;

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });

      // Process complete lines
      const lines = buf.split('\n');
      buf = lines.pop(); // keep incomplete tail

      for (const line of lines) {
        if (!line) continue;
        if (line.startsWith('[PIPELINE] ')) {
          onPipeline(line.slice(11));
        } else if (line.startsWith('[SOURCES] ')) {
          try { onSources(JSON.parse(line.slice(10))); } catch {}
        } else {
          onToken(line);
        }
      }
    }
    // Flush remaining buffer — process line by line (same logic as main loop)
    if (buf.trim()) {
      for (const line of buf.split('\n')) {
        if (!line.trim()) continue;
        if (line.startsWith('[PIPELINE] ')) {
          onPipeline(line.slice(11));
        } else if (line.startsWith('[SOURCES] ')) {
          try { onSources(JSON.parse(line.slice(10))); } catch {}
        } else {
          onToken(line);
        }
      }
    }
  }

  // ── Render: structured answer ──────────────────────────────
  function renderAnswer(container, raw) {
    const sourcesRe = /^sources[_ ]used\s*:/i;
    const lines = raw.split('\n').filter(l => l.trim() && !sourcesRe.test(l.trim()));
    if (!lines.length) return;

    const sopRe = /^for facility-specific procedures/i;
    const stepRe = /^\d+\.\s/;

    // First line = verdict
    const verdict = el('div', 'verdict');
    verdict.textContent = lines[0];
    container.appendChild(verdict);

    const steps = [];
    const body = [];
    let hasSop = false;

    for (let i = 1; i < lines.length; i++) {
      const l = lines[i].trim();
      if (stepRe.test(l)) steps.push(l.replace(/^\d+\.\s*/, ''));
      else if (sopRe.test(l)) hasSop = true;
      else body.push(l);
    }

    if (body.length) {
      const wrap = el('div', 'answer-body');
      body.forEach(line => {
        const p = el('p');
        p.textContent = line;
        wrap.appendChild(p);
      });
      container.appendChild(wrap);
    }

    if (steps.length) {
      const ol = el('ol', 'steps');
      steps.forEach(s => { const li = el('li'); li.textContent = s; ol.appendChild(li); });
      container.appendChild(ol);
    }

    if (hasSop) {
      const f = el('div', 'sop-footer');
      f.textContent = 'For facility-specific procedures (who to notify, where to quarantine, which forms to fill), consult your site SOP.';
      container.appendChild(f);
    }
  }

  // ── Render: source cards ───────────────────────────────────
  function renderSources(container, sources) {
    if (!sources || !sources.length) return;

    const wrap = el('div', 'sources');
    const label = el('div', 'sources-label');
    label.textContent = `Sources (${sources.length})`;
    wrap.appendChild(label);

    sources.forEach((src, i) => {
      const card = el('details', 'src-card');
      const sum = el('summary');

      const badge = el('span', 'src-badge');
      badge.textContent = i + 1;
      const num = el('span', 'src-num');
      num.textContent = src.section_number || '';
      const title = el('span', 'src-title');
      title.textContent = src.section_title || '';

      sum.appendChild(badge);
      sum.appendChild(num);
      sum.appendChild(title);
      card.appendChild(sum);

      if (src.full_text) {
        const body = el('div', 'src-body');
        body.textContent = src.full_text.length > 800
          ? src.full_text.slice(0, 800) + '\u2026'
          : src.full_text;
        card.appendChild(body);
      }
      wrap.appendChild(card);
    });

    container.appendChild(wrap);
    scroll();
  }

  // ── Send handler ───────────────────────────────────────────
  async function send() {
    if (busy) return;
    const text = $input.value.trim();
    if (!text) return;

    busy = true;
    $send.disabled = true;
    $input.value = '';
    resize();

    // Remove welcome on first message
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.remove();

    addUser(text);
    addSpinner();

    try {
      // Build assistant bubble
      const row = el('div', 'message assistant');
      const bubble = el('div', 'bubble');
      row.appendChild(bubble);

      // Pipeline trail (populated during streaming)
      const trail = el('div', 'pipeline-trail');
      bubble.appendChild(trail);

      // Streaming text area
      const textSpan = el('span', 'stream-text');
      bubble.appendChild(textSpan);

      // Cursor
      const cursor = el('span', 'cursor');
      bubble.appendChild(cursor);

      let rowShown = false;
      let accumulated = '';
      let streamSources = null;

      function showRow() {
        if (!rowShown) {
          removeSpinner();
          $messages.appendChild(row);
          rowShown = true;
        }
      }

      await stream(
        text,
        // onPipeline — show row on first step, add pill in real-time
        (step) => {
          showRow();
          // Mark previous step as done (yellow → green)
          const prev = trail.querySelector('.pip:last-child');
          if (prev) prev.classList.add('done');
          // Add new step as active (yellow)
          const s = el('span', 'pip');
          s.textContent = step;
          trail.appendChild(s);
          scroll();
        },
        // onToken
        (token) => {
          showRow();
          // Mark last pipeline step as done when answer starts
          const last = trail.querySelector('.pip:last-child:not(.done)');
          if (last) last.classList.add('done');
          accumulated += token;
          textSpan.textContent = accumulated;
          scroll();
        },
        // onSources
        (sources) => {
          streamSources = sources;
        }
      );

      // Edge case: nothing streamed at all
      showRow();

      // Remove cursor, replace raw text with structured rendering
      cursor.remove();
      textSpan.remove();
      renderAnswer(bubble, accumulated);

      // Show sources below the bubble
      if (streamSources) {
        renderSources(row, streamSources);
      }

      scroll();

    } catch (err) {
      removeSpinner();
      const row = el('div', 'message assistant');
      const b = el('div', 'bubble error');
      b.textContent = 'Error: ' + err.message;
      row.appendChild(b);
      $messages.appendChild(row);
    } finally {
      busy = false;
      $send.disabled = false;
      $input.focus();
    }
  }
})();
