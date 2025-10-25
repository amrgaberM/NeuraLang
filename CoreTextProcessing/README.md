# NeuraLang: CoreTextProcessing

This folder contains the **fundamental text processing steps** required for any NLP project. The goal is to transform raw text into a structured, numerical format that neural networks or statistical models can use effectively.

Text is inherently **unstructured and symbolic**, so we need a systematic way to represent it. Core text processing achieves this by:

1. **Cleaning and normalizing text** – removing noise like punctuation, lowercasing words, and handling special characters.
2. **Tokenizing text** – splitting sentences into individual units (words, subwords, or characters) that models can understand.
3. **Representing words numerically** – either as one-hot vectors or through co-occurrence matrices to capture relationships between words.

## Folder Structure and Purpose

- `tokenization_basics.ipynb`  
  Introduces **tokenization** concepts. Explains why tokenizing is critical for converting raw text into discrete elements. Covers:
  - Word-level tokenization
  - Handling punctuation and case
  - Examples for simple datasets

- `one_hot_encoding.ipynb`  
  Shows how to **represent each token as a unique vector** (one-hot encoding). This is the **first numerical representation** of text and helps understand embeddings:
  - Sparse vector representation
  - Precursor to dense embeddings like Word2Vec
  - Understanding limitations of one-hot vectors (high dimensionality, no semantic info)

- `cooccurrence_matrix.ipynb`  
  Builds **co-occurrence matrices** to capture how often words appear together in a context window.  
  This is **the statistical foundation for semantic embeddings**:
  - Rows = target words, columns = context words
  - Cell value = number of times words appear together
  - Demonstrates how co-occurrence encodes **word relationships**

## Key Concepts

- Text normalization and cleaning  
- Tokenization techniques  
- One-hot encoding as a baseline numeric representation  
- Co-occurrence statistics as a precursor to GloVe embeddings  

## Why This Matters

Understanding these core steps is crucial because **all NLP models, from Word2Vec to Transformers, rely on proper text representation**. Without this preprocessing, embeddings and neural models cannot capture meaningful patterns.

---