"""
Part 5: Build & Persist the FAISS Vector Database
----------------------------------------------------
FAISS (Facebook AI Similarity Search) stores our chunk embeddings in an
index optimized for fast nearest-neighbor search. We persist it to disk
(vector_db/) so we don't have to re-embed all documents every time the
app starts.

We store two things alongside each other:
1. index.faiss    -- the actual vector index (just numbers, no metadata)
2. metadata.pkl   -- a parallel list of {"source", "chunk_id", "text"}
                     dicts, so that when FAISS returns "vector #7 is the
                     closest match", we can look up what text that was.
"""

import os
import pickle
import numpy as np
import faiss

VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "metadata.pkl")


def build_index(embedded_chunks: list[dict]) -> faiss.Index:
    """
    Build a FAISS index from chunks that already have an "embedding" field
    (see embedder.embed_chunks). Uses flat L2 (Euclidean) distance search --
    simple and exact, ideal for small-to-medium document collections.
    """
    vectors = np.array([c["embedding"] for c in embedded_chunks]).astype("float32")
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index


def save_index(index: faiss.Index, embedded_chunks: list[dict]) -> None:
    """Persist the FAISS index and matching metadata (text minus embeddings) to disk."""
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    metadata = [
        {"source": c["source"], "chunk_id": c["chunk_id"], "text": c["text"]}
        for c in embedded_chunks
    ]
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[OK] Saved FAISS index ({index.ntotal} vectors) to {INDEX_PATH}")
    print(f"[OK] Saved metadata ({len(metadata)} entries) to {METADATA_PATH}")


def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load a previously persisted FAISS index + metadata from disk."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "No saved vector store found. Run this script directly first "
            "(python vector_store.py) to build one from data/."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def build_and_save_from_scratch() -> None:
    """Convenience pipeline: load -> chunk -> embed -> build index -> save."""
    from loader import load_documents
    from chunker import split_documents
    from embedder import embed_chunks

    docs = load_documents()
    chunks = split_documents(docs)
    embedded_chunks = embed_chunks(chunks)
    index = build_index(embedded_chunks)
    save_index(index, embedded_chunks)


if __name__ == "__main__":
    build_and_save_from_scratch()