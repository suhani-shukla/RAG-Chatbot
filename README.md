# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using your own documents instead of relying purely on an LLM's built-in knowledge.

## Project Structure

```
RAG_Chatbot/
│
├── data/               # Raw source documents (PDFs, text files) go here
├── vector_db/          # Persistent FAISS vector index files
├── app.py              # Main application entry point
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup Instructions

1. **Check Python version** (3.10+ required):
   ```bash
   python --version
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your documents**:
   Place any `.pdf` or `.txt` files you want the chatbot to learn from into the `data/` folder.
   A sample file (`data/sample.txt`) is included so you can test the pipeline immediately.

## Build Roadmap

- [ ] Part 1: Environment setup, project structure
- [ ] Part 2: Load and preprocess documents
- [ ] Part 3: Split documents into chunks
- [ ] Part 4: Generate vector embeddings
- [ ] Part 5: Build and persist the FAISS vector database
- [ ] Part 6: Test similarity search
- [ ] Part 7: Connect the retriever to an LLM
- [ ] Part 8: Build the full RAG chatbot engine
- [ ] Part 9: Build the Flask web interface

## Best Practices Followed

- Source citations are included in chatbot responses to reduce hallucination risk.
- Retrieval accuracy should be evaluated periodically using benchmark questions and user feedback.