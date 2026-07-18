"""
demo.py — Web app variant: answers ONLY from RAG_NOTES.txt, OFFLINE
------------------------------------------------------------------
Same two-pane UI as app.py (reuses templates/index.html and static/),
and you can still upload a document through the browser exactly like
in app.py. No matter what file you upload, the actual knowledge base
built for answering is always RAG_NOTES.txt.

OFFLINE MODE: this version does NOT call a live LLM. The full pipeline
(loading RAG_NOTES.txt, chunking, embedding, FAISS retrieval) all still
run for real -- only the final "generate an answer" step is replaced
with a predefined answer, looked up by matching the user's question
against a bank of Q&A pairs in canned_answers.py (using the same
embedding model as retrieval, so reworded questions still match).
This means the project can be demoed and tested without any API key
or Ollama installation.

IMPORTANT: This shares the same vector_db/ folder as app.py. Running
demo.py's upload will overwrite whatever index app.py built (and vice
versa), since they use the same underlying storage. Run one at a time,
or re-upload in app.py afterward to switch the shared index back to
your real document.

Run with:
    python demo.py

Then open http://127.0.0.1:5001 in your browser.
"""

import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

from loader import clean_text, load_txt
from chunker import split_documents
from embedder import embed_chunks
from vector_store import build_index, save_index
from retriever import retrieve
from canned_answers import find_canned_answer

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
NOTES_PATH = os.path.join(os.path.dirname(__file__), "RAG_NOTES.txt")
ALLOWED_EXTENSIONS = {"txt", "pdf"}
TOP_K = 3

os.makedirs(DATA_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_notes_index() -> int:
    """
    Load RAG_NOTES.txt specifically (ignores anything actually uploaded),
    then chunk, embed, and index it. Returns the number of chunks created.
    """
    if not os.path.exists(NOTES_PATH):
        raise FileNotFoundError(f"Could not find {NOTES_PATH}")

    raw_text = load_txt(NOTES_PATH)
    cleaned = clean_text(raw_text)

    if not cleaned:
        raise ValueError("RAG_NOTES.txt appears to be empty.")

    doc = {"source": "RAG_NOTES.txt", "text": cleaned}
    chunks = split_documents([doc])
    embedded_chunks = embed_chunks(chunks)

    index = build_index(embedded_chunks)
    save_index(index, embedded_chunks)

    return len(chunks)


def ask_offline(question: str, k: int = TOP_K) -> dict:
    """
    Same shape as rag_engine.ask(), but the generation step is replaced
    with a predefined answer instead of a live LLM call. Retrieval is
    still real -- this only swaps out the final response.
    """
    chunks = retrieve(question, k=k)
    answer = find_canned_answer(question)

    sources = [
        {"source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]}
        for c in chunks
    ]

    return {"question": question, "answer": answer, "sources": sources}


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
    file.save(filepath)  # saved for the record only; not used for answering

    try:
        num_chunks = build_notes_index()  # always indexes RAG_NOTES.txt instead
    except Exception as e:
        return jsonify({"error": f"Failed to process document: {e}"}), 500

    return jsonify({
        "filename": filename,
        "chunks": num_chunks,
        "note": "Demo mode: answers are grounded in RAG_NOTES.txt only.",
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    if not os.path.exists(INDEX_PATH):
        return jsonify({"error": "Upload any file first to trigger indexing of RAG_NOTES.txt."}), 503

    try:
        result = ask_offline(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)