"""
Part 9: Flask Web Interface with File Upload
-------------------------------------------------
Lets the user upload a .txt or .pdf document, then automatically runs
the full pipeline (load -> chunk -> embed -> build FAISS index) on it,
so the chatbot can immediately answer questions grounded in that document.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template_string

from loader import load_documents
from chunker import split_documents
from embedder import embed_chunks
from vector_store import build_index, save_index
from rag_engine import ask

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
ALLOWED_EXTENSIONS = {"txt", "pdf"}

os.makedirs(DATA_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RAG Chatbot</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; background: #f7f7f8; }
    h1 { font-size: 1.4rem; }
    #upload-box { background: white; border: 1px dashed #999; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    #status { font-size: 0.85rem; color: #555; margin-top: 8px; }
    #chat { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 250px; margin-bottom: 16px; }
    .msg { margin-bottom: 16px; }
    .msg .role { font-weight: 600; font-size: 0.85rem; color: #555; }
    .msg .text { margin-top: 4px; white-space: pre-wrap; }
    .sources { margin-top: 6px; font-size: 0.8rem; color: #777; }
    form.chat-form { display: flex; gap: 8px; }
    input[type=text] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 1rem; }
    button { padding: 10px 18px; border: none; border-radius: 6px; background: #2563eb; color: white; font-size: 1rem; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    button:disabled { background: #aaa; cursor: not-allowed; }
  </style>
</head>
<body>
  <h1>📄 RAG Chatbot</h1>

  <div id="upload-box">
    <input type="file" id="file-input" accept=".txt,.pdf">
    <button id="upload-btn">Upload &amp; Process</button>
    <div id="status">No document uploaded yet.</div>
  </div>

  <div id="chat"></div>
  <form class="chat-form" id="chat-form">
    <input type="text" id="question" placeholder="Ask a question about your document..." autocomplete="off" required disabled>
    <button type="submit" id="send-btn" disabled>Send</button>
  </form>

  <script>
    const chat = document.getElementById('chat');
    const form = document.getElementById('chat-form');
    const input = document.getElementById('question');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const status = document.getElementById('status');

    function addMessage(role, text, sources) {
      const div = document.createElement('div');
      div.className = 'msg';
      let sourcesHtml = '';
      if (sources && sources.length) {
        sourcesHtml = '<div class="sources">Sources: ' +
          sources.map(s => `${s.source} (chunk ${s.chunk_id})`).join(', ') +
          '</div>';
      }
      div.innerHTML = `<div class="role">${role}</div><div class="text">${text}</div>${sourcesHtml}`;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    uploadBtn.addEventListener('click', async () => {
      const file = fileInput.files[0];
      if (!file) {
        status.textContent = 'Please choose a file first.';
        return;
      }
      status.textContent = 'Uploading and processing document...';
      uploadBtn.disabled = true;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) {
          status.textContent = 'Error: ' + data.error;
        } else {
          status.textContent = `Ready! Indexed "${data.filename}" (${data.chunks} chunks). Ask away below.`;
          input.disabled = false;
          sendBtn.disabled = false;
        }
      } catch (err) {
        status.textContent = 'Upload failed: ' + err.message;
      } finally {
        uploadBtn.disabled = false;
      }
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const question = input.value.trim();
      if (!question) return;
      addMessage('You', question);
      input.value = '';

      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
        const data = await res.json();
        if (data.error) {
          addMessage('Error', data.error);
        } else {
          addMessage('Bot', data.answer, data.sources);
        }
      } catch (err) {
        addMessage('Error', 'Could not reach the server: ' + err.message);
      }
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .txt and .pdf files are supported."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(DATA_DIR, filename)
    file.save(filepath)

    try:
        docs = load_documents()
        chunks = split_documents(docs)
        embedded_chunks = embed_chunks(chunks)
        index = build_index(embedded_chunks)
        save_index(index, embedded_chunks)
    except Exception as e:
        return jsonify({"error": f"Failed to process document: {e}"}), 500

    return jsonify({"filename": filename, "chunks": len(chunks)})


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    if not os.path.exists(INDEX_PATH):
        return jsonify({"error": "Please upload a document first."}), 503

    try:
        result = ask(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
