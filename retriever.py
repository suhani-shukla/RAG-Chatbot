"""
Part 6: Similarity Search
-----------------------------
Given a user's query, embed it the same way we embedded chunks, then ask
FAISS for the k nearest chunk vectors. Those chunks are the most
semantically relevant pieces of our documents to the query.
"""

import numpy as np

from vector_store import load_index
from embedder import embed_query


def retrieve(query: str, k: int = 3) -> list[dict]:
    """
    Return the top-k most relevant chunks for a query, each annotated with
    a "score" (lower = more similar, since we use L2/Euclidean distance).
    """
    index, metadata = load_index()

    query_vector = np.array([embed_query(query)]).astype("float32")
    distances, indices = index.search(query_vector, k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:  # FAISS returns -1 if fewer than k vectors exist
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(dist)
        chunk["rank"] = rank + 1
        results.append(chunk)

    return results


if __name__ == "__main__":
    # Make sure a vector store already exists: run `python vector_store.py` first.
    test_queries = [
        "What is the admission fee?",
        "When is the application deadline?",
        "Tell me about scholarships",
    ]

    for q in test_queries:
        print(f"\nQuery: {q!r}")
        results = retrieve(q, k=2)
        for r in results:
            print(f"  #{r['rank']} (score={r['score']:.4f}) [{r['source']} chunk {r['chunk_id']}]")
            print(f"      {r['text'][:120]}...")