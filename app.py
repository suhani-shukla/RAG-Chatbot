"""
Part 9: Flask Web Interface (two-pane layout, upload + chat)
------------------------------------------------------------------
Serves the two-pane UI (templates/index.html + static/) and two API
endpoints:
  - POST /api/upload  -> save an uploaded document, run the full
                          pipeline (load -> chunk -> embed -> index)
  - POST /api/ask     -> run a question through the RAG engine

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

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


@app.route("/")
def index():
    return render_template("index.html")


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