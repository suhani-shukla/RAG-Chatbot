"""
Part 8: Complete RAG Chatbot Engine
---------------------------------------
Ties everything together:
  1. Retrieve the most relevant chunks for the user's question (Part 6).
  2. Construct a grounded prompt that includes those chunks as context.
  3. Send the prompt to the configured LLM (Part 7).
  4. Return the answer along with the sources it was grounded in.
"""
from retriever import retrieve
from llm import generate_answer

TOP_K = 3
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the context provided below.
If the answer cannot be found in the context, say "I don't have information about that in the provided documents."
Do not make up information that isn't in the context.
Context:
{context}
Question: {question}
Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_id']}]\n{c['text']}"
        for c in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def ask(question: str, k: int = TOP_K) -> dict:
    """
    Run the full RAG pipeline for a single question.
    Returns:
        {
            "question": "...",
            "answer": "...",
            "sources": [{"source": "sample.txt", "chunk_id": 0, "score": 0.12}, ...]
        }
    """
    chunks = retrieve(question, k=k)
    if not chunks:
        return {
            "question": question,
            "answer": "I don't have any documents loaded to answer questions from yet.",
            "sources": [],
        }
    prompt = build_prompt(question, chunks)

    try:
        answer = generate_answer(prompt)
    except Exception as e:
        answer = f"(LLM not available yet — retrieval worked, but skipped answer generation: {e})"

    sources = [
        {"source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]}
        for c in chunks
    ]
    return {"question": question, "answer": answer, "sources": sources}


if __name__ == "__main__":
    # Make sure a vector store already exists: run `python vector_store.py` first.
    result = ask("What is the admission fee?")
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_id']}, score={s['score']:.4f})")