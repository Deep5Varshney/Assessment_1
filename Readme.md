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
   
3. import re
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Hardcode 3-5 short documents
documents = [
    "Artificial Intelligence is transforming industries by automating complex workflows and data analysis.",
    "ChromaDB is an open-source vector database designed specifically for building AI applications with LLMs.",
    "Google Colab is a hosted Jupyter notebook service that requires no setup to use and provides free GPU access.",
    "Sentence Transformers allow developers to compute dense vector representations for sentences and images.",
]

# 2. Preprocess text (lowercase and remove punctuation)
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

cleaned_docs = [preprocess(doc) for doc in documents]

# 3. Initialize ChromaDB and Embedding Model
client = chromadb.Client()
collection = client.get_or_create_collection(name="mini_rag_pipeline")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(cleaned_docs).tolist()

# 4. Store embeddings and original documents in ChromaDB
ids = [f"doc_{i}" for i in range(len(documents))]
collection.add(
    documents=documents,  # Store original text for readable RAG output
    embeddings=embeddings,
    ids=ids,
)

# 5. Accept user query, embed, and retrieve the most relevant document
def retrieve_relevant_document(query_text):
    cleaned_query = preprocess(query_text)
    query_embedding = model.encode([cleaned_query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=1)

    return results["documents"][0][0]

# --- Test Run ---
user_query = "What is ChromaDB used for?"
print(f"Query: {user_query}")
print(f"Retrieved Response: {retrieve_relevant_document(user_query)}")
