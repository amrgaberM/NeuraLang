# NeuraLang: SemanticRepresentations

This folder focuses on **transforming processed text into dense, meaningful embeddings** that encode semantic and syntactic relationships between words. Unlike one-hot vectors, these embeddings capture **similarity and context**, enabling models to reason about language.

Word embeddings are essential because **neural networks cannot interpret symbolic text directly**. By mapping words to continuous vectors, we can:

- Measure semantic similarity between words  
- Solve analogy tasks (e.g., king – man + woman ≈ queen)  
- Provide rich input for downstream NLP models like RNNs, Transformers, or classifiers

## Folder Structure and Purpose

- `models.py`  
  Contains the final **Word2VecModel** and **GloVeModel** classes for reuse across experiments. Encapsulates training and evaluation logic.

- `word2vec_scratch.ipynb`  
  Implements **Skip-gram Word2Vec** from scratch. Demonstrates:
  - Forward and backward passes
  - Weight updates
  - Learning embeddings from local context
  - Illustrates how neural networks capture co-occurrence patterns

- `glove_from_counts.ipynb`  
  Implements **GloVe embeddings** using co-occurrence matrices. Focuses on:
  - Global statistical information
  - Matrix factorization to produce dense embeddings
  - Comparison with neural-based embeddings

- `embeddings_visualization.ipynb`  
  Visualizes embeddings in lower dimensions (PCA/t-SNE), showing **semantic clustering**:
  - Words with similar meaning appear close in the vector space
  - Useful for understanding and debugging embeddings

- `similarity_analogies_eval.ipynb`  
  Evaluates embeddings through **similarity metrics** and **analogy tests**:
  - Cosine similarity between word vectors
  - Classic analogy tasks to verify semantic relationships

## Key Concepts

- Word embeddings (Word2Vec and GloVe)  
- Capturing semantic relationships using dense vectors  
- Dimensionality reduction for visualization  
- Evaluating embeddings with similarity and analogy tasks

## Why This Matters

Dense embeddings are the **bridge between raw text and neural models**. Understanding how Word2Vec and GloVe work allows you to:

- Appreciate modern contextual embeddings (BERT, GPT)  
- Understand why pretraining improves downstream tasks  
- Experiment with embeddings in classification, generation, or similarity tasks

## Usage Flow

1. Take **preprocessed text** from `CoreTextProcessing/`.  
2. Train or load embedding models (Word2Vec or GloVe).  
3. Use embeddings for:
   - Similarity analysis  
   - Analogy solving  
   - Input to sequential models or Transformer architectures

---

