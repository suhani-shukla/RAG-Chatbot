"""
Part 3: Document Chunking
---------------------------
Splits loaded documents into smaller overlapping text chunks.

Why chunk?
- Embedding models and LLMs have limited context windows.
- Smaller, focused chunks retrieve more precisely than whole documents.

Why overlap?
- Prevents answers from being cut off at a chunk boundary.

Each chunk is returned as a dict:
    {
        "source": "sample.txt",
        "chunk_id": 0,
        "text": "chunk text..."
    }
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Take documents from loader.load_documents() and split each into chunks.
    Uses RecursiveCharacterTextSplitter, which tries to split on paragraph
    breaks first, then sentences, then words -- keeping chunks semantically
    coherent instead of cutting mid-sentence whenever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": piece,
            })

    return all_chunks


if __name__ == "__main__":
    from loader import load_documents

    docs = load_documents()
    chunks = split_documents(docs)

    print(f"\nTotal chunks created: {len(chunks)}")
    for c in chunks:
        print(f"- [{c['source']} #{c['chunk_id']}] ({len(c['text'])} chars): {c['text'][:100]}...")