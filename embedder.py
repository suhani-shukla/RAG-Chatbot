"""
Part 4: Generate Vector Embeddings
-------------------------------------
Converts each text chunk into a numeric vector ("embedding") that captures
its semantic meaning. Chunks with similar meaning end up with vectors that
are close together in vector space -- this is what makes similarity search
(Part 6) possible.

Model used: sentence-transformers/all-MiniLM-L6-v2
- Runs entirely locally (downloaded once, then cached) -- no API key needed.
- Small and fast (~80MB) while still giving solid retrieval quality.
- Produces 384-dimensional embedding vectors.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazy-loaded singleton so we don't reload the model repeatedly


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[INFO] Loading embedding model '{MODEL_NAME}' (first run downloads it)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Take chunks from chunker.split_documents() and attach an "embedding"
    field (a list of floats) to each one.
    """
    model = get_model()
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector

    return chunks


def embed_query(query: str):
    """Embed a single query string the same way chunks were embedded."""
    model = get_model()
    return model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]


if __name__ == "__main__":
    from loader import load_documents
    from chunker import split_documents

    docs = load_documents()
    chunks = split_documents(docs)
    chunks = embed_chunks(chunks)

    print(f"\nEmbedded {len(chunks)} chunks.")
    print(f"Embedding dimension: {len(chunks[0]['embedding'])}")
    print(f"Sample vector (first 5 values) for chunk 0: {chunks[0]['embedding'][:5]}")