const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const docCard = document.getElementById('doc-card');
const docName = document.getElementById('doc-name');
const docMeta = document.getElementById('doc-meta');
const replaceBtn = document.getElementById('replace-btn');
const statusLine = document.getElementById('status-line');

const chat = document.getElementById('chat');
const emptyState = document.getElementById('empty-state');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const sendBtn = document.getElementById('send-btn');

// ---- Upload handling ----

dropZone.addEventListener('click', () => fileInput.click());
replaceBtn.addEventListener('click', () => fileInput.click());

['dragover', 'dragenter'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  })
);

['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  })
);

dropZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) uploadFile(file);
});

async function uploadFile(file) {
  setStatus(`Processing "${file.name}"...`, false);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      setStatus(data.error || 'Upload failed.', true);
      return;
    }

    docName.textContent = data.filename;
    docMeta.textContent = `${data.chunks} chunks indexed`;
    docCard.hidden = false;
    dropZone.hidden = true;

    const groundingNote = data.note ? ` ${data.note}` : '';
    setStatus(`Ready.${groundingNote} Ask a question on the right.`, false);
    enableChat();
  } catch (err) {
    setStatus('Could not reach the server: ' + err.message, true);
  }
}

function setStatus(text, isError) {
  statusLine.textContent = text;
  statusLine.classList.toggle('error', !!isError);
}

function enableChat() {
  questionInput.disabled = false;
  sendBtn.disabled = false;
  if (emptyState) emptyState.remove();
  questionInput.focus();
}

// ---- Chat handling ----

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage('user', 'You', question);
  questionInput.value = '';
  sendBtn.disabled = true;

  const thinkingEl = addMessage('bot', 'Archive', '', [], true);

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();

    thinkingEl.remove();

    if (!res.ok || data.error) {
      addMessage('error', 'Error', data.error || 'Something went wrong.');
    } else {
      addMessage('bot', 'Archive', data.answer, data.sources);
    }
  } catch (err) {
    thinkingEl.remove();
    addMessage('error', 'Error', 'Could not reach the server: ' + err.message);
  } finally {
    sendBtn.disabled = false;
  }
});

function addMessage(kind, roleLabel, text, sources = [], isThinking = false) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${kind}${isThinking ? ' thinking' : ''}`;

  const role = document.createElement('div');
  role.className = 'msg-role';
  role.textContent = roleLabel;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  if (isThinking) {
    bubble.textContent = 'Thinking';
  } else if (kind === 'bot') {
    bubble.innerHTML = renderLiteMarkdown(text);
  } else {
    bubble.textContent = text;
  }

  wrap.appendChild(role);
  wrap.appendChild(bubble);

  if (sources && sources.length) {
    const sourcesEl = document.createElement('div');
    sourcesEl.className = 'sources';
    sources.forEach(s => {
      const card = document.createElement('span');
      card.className = 'source-card';
      card.textContent = `${s.source} · chunk ${s.chunk_id}`;
      sourcesEl.appendChild(card);
    });
    wrap.appendChild(sourcesEl);
  }

  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

// Minimal, safe markdown-lite renderer: escapes HTML first, then converts
// **bold**, *italic*, "• " bullets and "1. " numbered lines, and newlines.
// Deliberately small in scope -- not a full markdown parser.
function renderLiteMarkdown(raw) {
  const escapeHtml = (s) => s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  let html = escapeHtml(raw);

  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)\*(?!\*)/g, '<em>$1</em>');

  const lines = html.split('\n');
  const rendered = [];
  let inList = false;

  for (const line of lines) {
    const bulletMatch = line.match(/^\s*•\s+(.*)$/);
    const numberedMatch = line.match(/^\s*(\d+)\.\s+(.*)$/);

    if (bulletMatch || numberedMatch) {
      if (!inList) {
        rendered.push('<ul class="msg-list">');
        inList = true;
      }
      rendered.push(`<li>${bulletMatch ? bulletMatch[1] : numberedMatch[2]}</li>`);
    } else {
      if (inList) {
        rendered.push('</ul>');
        inList = false;
      }
      if (line.trim() === '') {
        rendered.push('<br>');
      } else {
        rendered.push(`<p class="msg-line">${line}</p>`);
      }
    }
  }
  if (inList) rendered.push('</ul>');

  return rendered.join('');
}