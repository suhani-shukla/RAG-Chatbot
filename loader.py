"""
Part 2: Document Loading & Preprocessing
------------------------------------------
Loads all supported files (.txt, .pdf) from the data/ folder,
extracts their raw text, and applies light preprocessing
(whitespace cleanup) so downstream chunking works cleanly.

Each loaded document is returned as a dict:
    {
        "source": "sample.txt",   # filename, used for citations later
        "text": "cleaned text..."
    }
"""

import os
import re
from pypdf import PdfReader

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def clean_text(text: str) -> str:
    """Collapse excess whitespace/newlines and strip leading/trailing space."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Scan data_dir for .txt and .pdf files, load and clean each one.
    Skips files that produce no extractable text and warns instead of crashing.
    """
    documents = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        ext = filename.lower().split(".")[-1]

        try:
            if ext == "txt":
                raw_text = load_txt(filepath)
            elif ext == "pdf":
                raw_text = load_pdf(filepath)
            else:
                continue

            cleaned = clean_text(raw_text)

            if not cleaned:
                print(f"[WARNING] No extractable text in '{filename}', skipping.")
                continue

            documents.append({"source": filename, "text": cleaned})
            print(f"[OK] Loaded '{filename}' ({len(cleaned)} characters)")

        except Exception as e:
            print(f"[ERROR] Failed to load '{filename}': {e}")

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"\nTotal documents loaded: {len(docs)}")
    for doc in docs:
        preview = doc["text"][:150].replace("\n", " ")
        print(f"- {doc['source']}: {preview}...")