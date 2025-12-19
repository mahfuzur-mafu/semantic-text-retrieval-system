# semantic-text-retrieval-system


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-all--MiniLM--L6--v2-orange)](https://www.sbert.net/)
[![Hugging Face Datasets](https://img.shields.io/badge/Hugging_Face-Datasets-green)](https://huggingface.co/datasets)

##Overview

This project implements an end to end semantic text retrieval system based on sentence embeddings and vector similarity.
Instead of relying on keyword matching, the system retrieves semantically similar text by comparing vector representations of text chunks and user prompts.

The system is designed to be reusable across multiple applications such as semantic search, recommendation systems, and content discovery.

## Key Idea

Text is first converted into fixed length numerical vectors using a pretrained embedding model.
These vectors capture semantic meaning.
At query time, a user prompt is embedded into the same vector space and compared against stored vectors using similarity metrics.

## System Architecture

### Offline embedding pipeline

1. Load dataset from Hugging Face

2. Select and clean the answer column

3. Split text into small chunks

4. Convert chunks into dense embeddings using a sentence embedding model

5. Store embeddings and corresponding text locally

## How It Works

### Online retrieval pipeline

1. User provides a text prompt

2. Prompt is converted into an embedding

3. Similarity is computed against stored embeddings

4. Top matching n text chunks are returned

Results are displayed through a web interface

## Embedding Models Used

The system was tested with multiple embedding models to study the effect of dimensionality and performance.

1. all MiniLM L6 v2 with 384 dimensional embeddings

This allowed comparison of speed, memory usage, and retrieval quality across different embedding sizes.

## Similarity Metrics

Two similarity measures are used.

### Euclidean distance
Lower distance indicates higher similarity

### Cosine similarity
Higher score indicates higher similarity

Both metrics are computed and shown for each query to better understand retrieval behavior.

## Web Interface

A lightweight web interface allows users to interact with the system.

1. Prompt based semantic search

2. Configurable top K results

3. Clear display of closest matches using Euclidean and cosine similarity

4. Display of embedding dimension and model information

5. Clean separation of frontend and backend logic

## Technology Stack
### Backend

1. Python

2. Sentence Transformers

3. NumPy

4. Hugging Face Datasets and model

### Frontend

1. HTML

2. CSS

3. JavaScript

### Node based serving

1. Express

2. Transformers.js for embedding inference

3. Local static file serving

## Use Cases

This system can be extended to:

1. Semantic document search

2. Question answer retrieval

3. Recommendation systems

4. RAG style pipelines

5. Content similarity and clusterin
