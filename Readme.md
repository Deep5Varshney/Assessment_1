# Mini RAG Pipeline with ChromaDB

A lightweight Retrieval-Augmented Generation (RAG) pipeline built and executed in Google Colab using Python, `sentence-transformers`, and `ChromaDB`.

## Features
- **Text Preprocessing:** Converts text to lowercase and removes punctuation using regular expressions.
- **Embeddings:** Computes dense vector representations using Hugging Face's `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Vector Database:** Stores and queries document embeddings efficiently using `ChromaDB`.
- **Semantic Retrieval:** Accepts user queries and retrieves the most relevant document based on vector similarity.

## Tech Stack
- Python
- Google Colab
- ChromaDB
- Sentence Transformers

## How to Run
1. Open Google Colab and create a new notebook.
2. Install dependencies:
   ```bash
   pip install chromadb sentence-transformers
