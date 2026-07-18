"""
canned_answers.py — Predefined answers for offline demo mode
------------------------------------------------------------------
Used by demo.py to replace live LLM generation with predefined answers,
so the project can be demonstrated/tested without a working API key.

Two layers of matching:
  1. Greetings / small talk ("hi", "how are you", etc.) are matched with
     simple pattern matching -- these are too short and generic for
     embedding similarity to handle reliably, and they aren't really
     "content" from RAG_NOTES.txt anyway.
  2. Everything else is matched against a bank of predefined Q&A pairs
     using the same embedding model already used for chunk retrieval
     (embedder.embed_query) via cosine similarity, so close rewordings
     of a predefined question (not just exact matches) still work.

If nothing matches closely enough, a fallback message lists the topics
that do have predefined answers -- styled the same conversational way
as a real answer, not a bare error string.
"""

import re
import numpy as np

from embedder import embed_query

SIMILARITY_THRESHOLD = 0.45  # cosine similarity; below this, use the fallback

# ---- Layer 1: greetings / small talk ----

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|hiya|yo|good (morning|afternoon|evening))[\s!.,]*$", re.IGNORECASE)
HOW_ARE_YOU_RE = re.compile(r"how('?s| is| are)\s+(you|it going|things)|what'?s up", re.IGNORECASE)
THANKS_RE = re.compile(r"^\s*(thanks|thank you|thx|cheers)[\s!.,]*$", re.IGNORECASE)

GREETING_ANSWER = (
    "Hey there! 👋 I'm running in offline demo mode, so I'm answering using a set of "
    "predefined responses grounded in **RAG_NOTES.txt** — no live API calls involved.\n\n"
    "Ask me something like:\n"
    "• \"What is RAG?\"\n"
    "• \"How does chunking work?\"\n"
    "• \"What are the limitations of RAG?\"\n\n"
    "Go ahead, ask away!"
)

HOW_ARE_YOU_ANSWER = (
    "Doing well, thanks for asking! I'm a demo assistant grounded in RAG_NOTES.txt, so "
    "I don't have real feelings — but I *do* have opinions about vector databases if "
    "you're interested. 😄\n\n"
    "What would you like to know about RAG?"
)

THANKS_ANSWER = (
    "You're welcome! Let me know if there's anything else about RAG you'd like to explore."
)


def _match_small_talk(question: str) -> str | None:
    q = question.strip()
    if GREETING_RE.match(q):
        return GREETING_ANSWER
    if HOW_ARE_YOU_RE.search(q):
        return HOW_ARE_YOU_ANSWER
    if THANKS_RE.match(q):
        return THANKS_ANSWER
    return None


# ---- Layer 2: predefined Q&A about RAG_NOTES.txt content ----

QA_PAIRS = [
    {
        "question": "What is RAG?",
        "answer": (
            "Great question — here's the core idea:\n\n"
            "**Retrieval-Augmented Generation (RAG)** combines a retrieval system with a "
            "large language model. Instead of answering purely from what it memorized "
            "during training, the LLM first retrieves relevant information from a "
            "knowledge base, then uses that as context to generate a grounded answer.\n\n"
            "In short: *retrieve first, then generate.* Feel free to ask anything else!"
        ),
    },
    {
        "question": "Why does RAG exist? What problem does it solve?",
        "answer": (
            "Here's why RAG matters — it addresses three core LLM limitations:\n\n"
            "• **Knowledge cutoff** — models don't know about data after their training "
            "date, or private data they were never trained on.\n"
            "• **Hallucination** — models can generate plausible but false information; "
            "grounding answers in retrieved text helps reduce this.\n"
            "• **Domain specificity** — general models may lack depth in a narrow domain, "
            "which RAG lets them \"read\" on demand.\n\n"
            "Let me know if you'd like more detail on any of these!"
        ),
    },
    {
        "question": "What are the stages of a RAG pipeline?",
        "answer": (
            "Here's the full pipeline, stage by stage:\n\n"
            "1. Ingestion\n"
            "2. Preprocessing\n"
            "3. Chunking\n"
            "4. Embedding\n"
            "5. Vector storage\n"
            "6. Retrieval\n"
            "7. Reranking (optional)\n"
            "8. Prompt construction\n"
            "9. Generation\n"
            "10. Citation (optional, but recommended)\n\n"
            "Want me to go deeper on any one of these steps?"
        ),
    },
    {
        "question": "What is chunking and why is it needed?",
        "answer": (
            "Good one — here's the deal with chunking:\n\n"
            "**Chunking** is splitting documents into smaller passages before embedding "
            "them. It's needed because embedding models and LLMs have limited context "
            "windows, and smaller, focused chunks retrieve more precisely than embedding "
            "a whole document as a single vector.\n\n"
            "Curious about specific chunking strategies? Just ask!"
        ),
    },
    {
        "question": "What chunking strategies are there?",
        "answer": (
            "Here are the main chunking strategies:\n\n"
            "• **Fixed-size chunking** — split every N characters/tokens, often with "
            "overlap so a sentence spanning a boundary isn't cut off.\n"
            "• **Recursive/structure-aware chunking** — split on paragraphs first, then "
            "sentences, then words, keeping chunks more coherent.\n"
            "• **Semantic chunking** — split based on topic shifts detected via "
            "embeddings.\n"
            "• **Document-aware chunking** — respects headers, sections, and table "
            "boundaries.\n\n"
            "Feel free to ask anything else!"
        ),
    },
    {
        "question": "What embedding models are commonly used?",
        "answer": (
            "Here's a breakdown of common embedding model choices:\n\n"
            "• **Open-source, local models** — Sentence Transformers (e.g. "
            "all-MiniLM-L6-v2), BGE, GTE. No API needed, no per-request cost.\n"
            "• **Hosted API models** — OpenAI's, Google's, and Cohere's embedding "
            "models. Typically higher quality, but need an API key and cost per call.\n\n"
            "Anything else you'd like to dig into?"
        ),
    },
    {
        "question": "What vector stores or vector databases are commonly used?",
        "answer": (
            "Here's the landscape of vector stores:\n\n"
            "• **FAISS** — a library for efficient local similarity search, great for "
            "small-to-medium datasets.\n"
            "• **Chroma** — an open-source vector database, good for prototyping.\n"
            "• **Pinecone / Weaviate / Qdrant** — managed or self-hosted databases built "
            "for production scale, with metadata filtering and hybrid search.\n\n"
            "Let me know if you want a comparison of any of these!"
        ),
    },
    {
        "question": "How can retrieval quality be improved?",
        "answer": (
            "Here's how retrieval quality typically gets improved:\n\n"
            "• **Top-k tuning** — too few risks missing context, too many dilutes the "
            "prompt.\n"
            "• **Hybrid search** — combining keyword search (like BM25) with vector "
            "similarity, since exact matches (product codes, names) can be missed by "
            "pure semantic search.\n"
            "• **Reranking** — using a cross-encoder or LLM-based reranker to reorder "
            "retrieved chunks by true relevance before generation.\n\n"
            "Want more detail on any of these?"
        ),
    },
    {
        "question": "What are best practices for prompt construction in RAG?",
        "answer": (
            "Here are the key prompt construction best practices:\n\n"
            "• Explicitly instruct the model to answer **only** from the provided "
            "context, and to say when it doesn't know.\n"
            "• Include source labels (filename, page, chunk ID) so the model can cite "
            "them in its answer.\n"
            "• Keep context organized, avoiding unnecessary duplication across chunks.\n\n"
            "Ask away if you want examples!"
        ),
    },
    {
        "question": "What are common applications of RAG?",
        "answer": (
            "RAG shows up in quite a few real-world use cases:\n\n"
            "• Customer support assistants grounded in product documentation\n"
            "• Legal and compliance document search\n"
            "• Healthcare information assistants referencing clinical guidelines\n"
            "• Enterprise knowledge bases (internal wikis, policies, past tickets)\n"
            "• Research assistants searching papers or technical documentation\n"
            "• Financial advisory tools referencing filings and disclosures\n\n"
            "Curious how any of these actually get built? Just ask!"
        ),
    },
    {
        "question": "What are the limitations of RAG?",
        "answer": (
            "Here's the honest list of RAG's limitations:\n\n"
            "• **Garbage in, garbage out** — retrieval quality depends entirely on the "
            "underlying documents being accurate and well-chunked.\n"
            "• **Retrieval mismatch** — a query phrased very differently from the source "
            "text may fail to retrieve relevant chunks.\n"
            "• **Context window limits** — too many/large chunks can overwhelm the LLM.\n"
            "• **Added latency** — the multi-stage pipeline is slower than a direct LLM "
            "call.\n"
            "• **Still can hallucinate** — RAG reduces, but doesn't eliminate, "
            "hallucination risk.\n\n"
            "Feel free to ask anything else!"
        ),
    },
    {
        "question": "How do you evaluate a RAG system?",
        "answer": (
            "Here's how RAG systems typically get evaluated:\n\n"
            "• **Retrieval metrics** — precision/recall of retrieved chunks against "
            "benchmark questions with known correct passages.\n"
            "• **Answer quality metrics** — human evaluation or LLM-as-judge scoring for "
            "faithfulness and relevance.\n"
            "• **End-to-end feedback loops** — logging real user questions and outcomes "
            "(thumbs up/down) to spot weak points over time.\n\n"
            "Want to know more about any of these approaches?"
        ),
    },
    {
        "question": "How does RAG compare to fine-tuning?",
        "answer": (
            "Good comparison to make — here's how they differ:\n\n"
            "• **Fine-tuning** bakes new knowledge or behavior into model weights "
            "through additional training. Great for teaching a new style or task "
            "format, but expensive to update and doesn't scale well for fast-changing "
            "facts.\n"
            "• **RAG** keeps the model's weights unchanged and supplies fresh, "
            "swappable knowledge at query time — easier to update (just change the "
            "documents) and more transparent (answers can cite sources).\n\n"
            "In practice, they're often complementary: fine-tune for tone/instruction-"
            "following, use RAG for factual grounding. Anything else you'd like to know?"
        ),
    },
]

FALLBACK_MESSAGE = (
    "Hmm, I don't have a predefined answer for that one in demo mode — I'm limited to "
    "a fixed set of topics here rather than free-form generation.\n\n"
    "Here's what I *can* help with:\n"
    "• What RAG is, and why it exists\n"
    "• The RAG pipeline stages\n"
    "• Chunking strategies\n"
    "• Embedding models and vector stores\n"
    "• Retrieval quality and prompt construction\n"
    "• RAG's applications and limitations\n"
    "• Evaluating a RAG system\n"
    "• RAG vs. fine-tuning\n\n"
    "Try rephrasing your question around one of these!"
)

_qa_embeddings = None  # lazy-computed once, cached for the process lifetime


def _get_qa_embeddings():
    global _qa_embeddings
    if _qa_embeddings is None:
        _qa_embeddings = np.array([embed_query(qa["question"]) for qa in QA_PAIRS])
    return _qa_embeddings


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def find_canned_answer(question: str) -> str:
    """
    First checks for greetings/small talk (pattern-matched, not embedded).
    Otherwise, embeds the question and compares it against all predefined
    questions using cosine similarity, returning the closest match's
    answer if it's similar enough, or a styled fallback otherwise.
    """
    small_talk = _match_small_talk(question)
    if small_talk is not None:
        return small_talk

    query_vector = embed_query(question)
    qa_vectors = _get_qa_embeddings()

    similarities = [_cosine_similarity(query_vector, v) for v in qa_vectors]
    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        return QA_PAIRS[best_idx]["answer"]

    return FALLBACK_MESSAGE