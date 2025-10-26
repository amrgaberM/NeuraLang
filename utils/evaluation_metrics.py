"""
Evaluation metrics for NeuraLang project.
Functions for computing similarity, accuracy, and other evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict


# ============================================================================
# SIMILARITY METRICS
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 0.0
    
    return dot_product / norm_product


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Euclidean distance (lower is more similar)
    """
    return np.linalg.norm(vec1 - vec2)


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Manhattan (L1) distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Manhattan distance (lower is more similar)
    """
    return np.sum(np.abs(vec1 - vec2))


def compute_similarity(word1: str, word2: str, 
                      embeddings_dict: Dict[str, np.ndarray],
                      metric: str = 'cosine') -> Optional[float]:
    """
    Compute similarity between two words using their embeddings.
    
    Args:
        word1: First word
        word2: Second word
        embeddings_dict: Dictionary mapping words to embedding vectors
        metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
    
    Returns:
        Similarity score, or None if words not found
    """
    if word1 not in embeddings_dict or word2 not in embeddings_dict:
        return None
    
    vec1 = embeddings_dict[word1]
    vec2 = embeddings_dict[word2]
    
    if metric == 'cosine':
        return cosine_similarity(vec1, vec2)
    elif metric == 'euclidean':
        return -euclidean_distance(vec1, vec2)  # Negate so higher is better
    elif metric == 'manhattan':
        return -manhattan_distance(vec1, vec2)  # Negate so higher is better
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# WORD SIMILARITY EVALUATION
# ============================================================================

def evaluate_on_similarity_dataset(
    embeddings_dict: Dict[str, np.ndarray],
    dataset: List[Tuple[str, str, float]],
    metric: str = 'cosine'
) -> Dict:
    """
    Evaluate embeddings on a word similarity dataset.
    
    Args:
        embeddings_dict: Dictionary of word embeddings
        dataset: List of (word1, word2, human_score) tuples
        metric: Similarity metric to use
    
    Returns:
        Dictionary with evaluation results including correlations
    """
    model_scores = []
    human_scores = []
    missing_pairs = 0
    
    for word1, word2, human_score in dataset:
        sim = compute_similarity(word1, word2, embeddings_dict, metric)
        if sim is not None:
            model_scores.append(sim)
            human_scores.append(human_score)
        else:
            missing_pairs += 1
    
    if len(model_scores) < 2:
        return {
            'spearman': 0.0,
            'pearson': 0.0,
            'coverage': 0.0,
            'num_pairs': 0,
            'missing_pairs': missing_pairs
        }
    
    # Compute correlations
    spearman_corr, spearman_pval = spearmanr(human_scores, model_scores)
    pearson_corr, pearson_pval = pearsonr(human_scores, model_scores)
    
    return {
        'spearman': spearman_corr,
        'spearman_pval': spearman_pval,
        'pearson': pearson_corr,
        'pearson_pval': pearson_pval,
        'coverage': (len(model_scores) / len(dataset)) * 100,
        'num_pairs': len(model_scores),
        'missing_pairs': missing_pairs,
        'model_scores': model_scores,
        'human_scores': human_scores
    }


# ============================================================================
# WORD ANALOGY EVALUATION
# ============================================================================

def solve_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    embeddings_dict: Dict[str, np.ndarray],
    top_k: int = 5,
    exclude_input_words: bool = True
) -> List[Tuple[str, float]]:
    """
    Solve word analogy: A is to B as C is to ?
    
    Uses vector arithmetic: vec(B) - vec(A) + vec(C) ≈ vec(D)
    
    Args:
        word_a, word_b, word_c: The three given words
        embeddings_dict: Dictionary of word embeddings
        top_k: Number of candidate answers to return
        exclude_input_words: Whether to exclude input words from results
    
    Returns:
        List of (word, similarity_score) tuples
    """
    # Check if all words are in vocabulary
    if not all(w in embeddings_dict for w in [word_a, word_b, word_c]):
        return []
    
    # Compute target vector: B - A + C
    vec_a = embeddings_dict[word_a]
    vec_b = embeddings_dict[word_b]
    vec_c = embeddings_dict[word_c]
    
    target_vec = vec_b - vec_a + vec_c
    
    # Find most similar words to target vector
    similarities = []
    exclude_words = {word_a, word_b, word_c} if exclude_input_words else set()
    
    for word, vec in embeddings_dict.items():
        if word not in exclude_words:
            sim = cosine_similarity(target_vec, vec)
            similarities.append((word, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def evaluate_on_analogies(
    embeddings_dict: Dict[str, np.ndarray],
    dataset: List[Tuple[str, str, str, str, str]],
    top_k: int = 1
) -> Dict:
    """
    Evaluate embeddings on word analogy dataset.
    
    Args:
        embeddings_dict: Dictionary of word embeddings
        dataset: List of (word_a, word_b, word_c, expected_d, category) tuples
        top_k: Consider top-k predictions as correct
    
    Returns:
        Dictionary with overall and per-category accuracy
    """
    category_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct = 0
    total_evaluated = 0
    skipped = 0
    
    for word_a, word_b, word_c, expected_d, category in dataset:
        # Check if all words are in vocabulary
        if not all(w in embeddings_dict for w in [word_a, word_b, word_c, expected_d]):
            skipped += 1
            continue
        
        # Solve analogy
        predictions = solve_analogy(word_a, word_b, word_c, embeddings_dict, top_k=top_k)
        
        if not predictions:
            skipped += 1
            continue
        
        # Check if expected answer is in top-k predictions
        predicted_words = [w for w, _ in predictions]
        is_correct = expected_d in predicted_words
        
        category_results[category]['total'] += 1
        total_evaluated += 1
        
        if is_correct:
            category_results[category]['correct'] += 1
            total_correct += 1
    
    # Compute accuracies
    overall_accuracy = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
    
    category_accuracies = {}
    for category, stats in category_results.items():
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        category_accuracies[category] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total']
        }
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_evaluated': total_evaluated,
        'skipped': skipped,
        'coverage': (total_evaluated / len(dataset) * 100) if len(dataset) > 0 else 0,
        'category_accuracies': category_accuracies
    }


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy score between 0 and 1
    """
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                        average: str = 'binary') -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary' for binary classification, 'macro' for multiclass
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    elif average == 'macro':
        classes = np.unique(y_true)
        precisions = []
        recalls = []
        f1s = []
        
        for cls in classes:
            y_true_cls = (y_true == cls).astype(int)
            y_pred_cls = (y_pred == cls).astype(int)
            
            p, r, f = precision_recall_f1(y_true_cls, y_pred_cls, average='binary')
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)
    
    else:
        raise ValueError(f"Unknown average type: {average}")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                    num_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes (auto-detected if None)
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    return cm


# ============================================================================
# SEQUENCE METRICS
# ============================================================================

def perplexity(log_probs: np.ndarray) -> float:
    """
    Compute perplexity from log probabilities.
    
    Perplexity = exp(-1/N * sum(log P(x_i)))
    
    Args:
        log_probs: Array of log probabilities
    
    Returns:
        Perplexity score (lower is better)
    """
    return np.exp(-np.mean(log_probs))


def bleu_score(reference: List[str], candidate: List[str], n: int = 4) -> float:
    """
    Compute BLEU score for sequence evaluation.
    
    Simplified implementation of BLEU-n score.
    
    Args:
        reference: Reference sequence (list of tokens)
        candidate: Candidate sequence (list of tokens)
        n: Maximum n-gram size
    
    Returns:
        BLEU score between 0 and 1
    """
    from collections import Counter
    
    # Brevity penalty
    bp = 1.0 if len(candidate) >= len(reference) else np.exp(1 - len(reference) / len(candidate))
    
    # Compute n-gram precisions
    precisions = []
    
    for i in range(1, n + 1):
        # Generate n-grams
        ref_ngrams = Counter([tuple(reference[j:j+i]) for j in range(len(reference) - i + 1)])
        cand_ngrams = Counter([tuple(candidate[j:j+i]) for j in range(len(candidate) - i + 1)])
        
        # Count matches
        matches = sum((cand_ngrams & ref_ngrams).values())
        total = sum(cand_ngrams.values())
        
        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geo_mean = 0.0
    
    return bp * geo_mean


# ============================================================================
# NEAREST NEIGHBORS
# ============================================================================

def find_nearest_neighbors(
    query_word: str,
    embeddings_dict: Dict[str, np.ndarray],
    top_k: int = 10,
    metric: str = 'cosine'
) -> List[Tuple[str, float]]:
    """
    Find nearest neighbors of a word in embedding space.
    
    Args:
        query_word: Word to find neighbors for
        embeddings_dict: Dictionary of word embeddings
        top_k: Number of neighbors to return
        metric: Distance metric ('cosine', 'euclidean', 'manhattan')
    
    Returns:
        List of (word, similarity_score) tuples
    """
    if query_word not in embeddings_dict:
        return []
    
    query_vec = embeddings_dict[query_word]
    similarities = []
    
    for word, vec in embeddings_dict.items():
        if word != query_word:
            if metric == 'cosine':
                sim = cosine_similarity(query_vec, vec)
            elif metric == 'euclidean':
                sim = -euclidean_distance(query_vec, vec)
            elif metric == 'manhattan':
                sim = -manhattan_distance(query_vec, vec)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            similarities.append((word, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


# ============================================================================
# EMBEDDING QUALITY METRICS
# ============================================================================

def embedding_coherence(embeddings: np.ndarray) -> float:
    """
    Measure coherence/quality of embeddings using average pairwise cosine similarity.
    
    Args:
        embeddings: Matrix of embeddings (num_words, embedding_dim)
    
    Returns:
        Average pairwise cosine similarity
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    # Compute pairwise similarities
    similarity_matrix = np.dot(normalized, normalized.T)
    
    # Average excluding diagonal
    total_sim = np.sum(similarity_matrix) - n  # Subtract diagonal (self-similarities)
    avg_sim = total_sim / (n * (n - 1))
    
    return avg_sim


def embedding_coverage(embeddings_dict: Dict[str, np.ndarray], 
                      test_words: List[str]) -> float:
    """
    Compute vocabulary coverage for a list of test words.
    
    Args:
        embeddings_dict: Dictionary of word embeddings
        test_words: List of words to check coverage for
    
    Returns:
        Coverage percentage (0-100)
    """
    if not test_words:
        return 0.0
    
    covered = sum(1 for word in test_words if word in embeddings_dict)
    return (covered / len(test_words)) * 100