# semantic-text-retrieval-system

# Semantic Text Retrieval System with Vector Search

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-all--MiniLM--L6--v2-orange)](https://www.sbert.net/)
[![Hugging Face Datasets](https://img.shields.io/badge/Hugging_Face-Datasets-green)](https://huggingface.co/datasets)

This project implements a complete **semantic text retrieval system** using dense vector embeddings and similarity search. It demonstrates how to build an efficient semantic search engine without relying on traditional keyword matching.

Instead of exact word overlap, the system retrieves the most **semantically relevant** text chunks by comparing high-dimensional vector representations of text.

Perfect for:
- Semantic search engines
- Question-Answering retrieval
- Retrieval-Augmented Generation (RAG) pipelines
- Recommendation systems
- Content discovery and clustering

## Key Features

- Loads real-world Q&A data from Hugging Face (`sentence-transformers/gooaq`)
- Splits long answers into meaningful chunks
- Generates dense embeddings using **Sentence Transformers**
- Performs **exact nearest neighbor search** using Euclidean distance and Cosine similarity
- Compares both similarity metrics side-by-side
- Saves embeddings, model, and text chunks locally for fast reuse (e.g., in web apps)
- Fully reproducible Jupyter notebook (`final_e.ipynb`)

## How It Works

### Offline Indexing Pipeline
1. Load dataset from Hugging Face
2. Extract and clean answer text
3. Chunk text into small segments (~60 characters, word-aware)
4. Encode all chunks into dense vectors using a pretrained model
5. Store vectors and text in a simple **in-memory vector database**

### Online Retrieval Pipeline
1. User provides a query (e.g., *"What is the capital of Finland?"*)
2. Query is encoded into the same vector space
3. Compute similarity against all stored vectors
4. Return top-k most relevant text chunks
5. Display results with both **Euclidean distance** and **Cosine similarity**

## Embedding Model Used

```text
sentence-transformers/all-MiniLM-L6-v2
