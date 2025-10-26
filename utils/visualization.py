"""
Visualization utilities for NeuraLang project.
Functions for plotting embeddings, attention mechanisms, and evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns


# ============================================================================
# EMBEDDING VISUALIZATION FUNCTIONS
# ============================================================================

def plot_embeddings_2d(
    embeddings: np.ndarray,
    words: List[str],
    method: str = 'pca',
    perplexity: int = 30,
    n_iter: int = 1000,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    highlight_words: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize word embeddings in 2D using PCA or t-SNE.
    Compatible with all sklearn versions.
    """
    import inspect
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
        if title is None:
            title = f'Word Embeddings Visualization (PCA)\nExplained Variance: {reducer.explained_variance_ratio_.sum():.2%}'

    elif method.lower() == 'tsne':
        # Handle compatibility: sklearn >=1.6 uses max_iter instead of n_iter
        tsne_params = {
            'n_components': 2,
            'perplexity': perplexity,
            'random_state': 42
        }
        if 'max_iter' in inspect.signature(TSNE).parameters:
            tsne_params['max_iter'] = n_iter
        else:
            tsne_params['n_iter'] = n_iter

        reducer = TSNE(**tsne_params)
        coords_2d = reducer.fit_transform(embeddings)

        if title is None:
            title = f'Word Embeddings Visualization (t-SNE)\nPerplexity: {perplexity}'

    else:
        raise ValueError(f"Method must be 'pca' or 'tsne', got {method}")

    # Plot setup
    fig, ax = plt.subplots(figsize=figsize)

    # Highlight logic
    if highlight_words:
        highlight_indices = [i for i, w in enumerate(words) if w in highlight_words]
        normal_indices = [i for i in range(len(words)) if i not in highlight_indices]

        if normal_indices:
            ax.scatter(coords_2d[normal_indices, 0], coords_2d[normal_indices, 1],
                       alpha=0.5, s=30, c='steelblue', label='Other words')

        if highlight_indices:
            ax.scatter(coords_2d[highlight_indices, 0], coords_2d[highlight_indices, 1],
                       alpha=0.8, s=100, c='red', marker='*', label='Highlighted words')
    else:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.6, s=50, c='steelblue')

    # Add labels
    for i, word in enumerate(words):
        color = 'red' if highlight_words and word in highlight_words else 'black'
        fontsize = 10 if highlight_words and word in highlight_words else 8
        fontweight = 'bold' if highlight_words and word in highlight_words else 'normal'
        ax.annotate(word, (coords_2d[i, 0], coords_2d[i, 1]),
                    fontsize=fontsize, fontweight=fontweight, color=color,
                    xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=11)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if highlight_words:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

def plot_semantic_clusters(
    embeddings: np.ndarray,
    words: List[str],
    categories: Dict[str, List[str]],
    method: str = 'tsne',
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize semantic clusters of word embeddings with PCA or t-SNE.
    Fully compatible with all scikit-learn versions (handles n_iter / max_iter).
    
    Args:
        embeddings: Array of shape (n_words, embedding_dim)
        words: List of words corresponding to embeddings
        categories: Dict mapping category names to lists of words
        method: 'pca' or 'tsne' for dimensionality reduction
        figsize: Figure size (width, height)
        save_path: Path to save the generated figure (optional)
    
    Returns:
        matplotlib Figure object
    """
    import inspect
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # === Build mapping from word to index ===
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Collect indices per semantic group
    category_indices = {
        cat_name: [word_to_idx[w] for w in cat_words if w in word_to_idx]
        for cat_name, cat_words in categories.items()
    }

    # === Dimensionality Reduction ===
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)

    elif method.lower() == 'tsne':
        # Safe t-SNE initialization for all sklearn versions
        tsne_args = {
            'n_components': 2,
            'perplexity': 30,
            'random_state': 42
        }

        # Detect correct iteration parameter
        tsne_sig = inspect.signature(TSNE.__init__).parameters
        if 'max_iter' in tsne_sig:          # sklearn >= 1.6
            tsne_args['max_iter'] = 1000
        elif 'n_iter' in tsne_sig:          # sklearn < 1.6
            tsne_args['n_iter'] = 1000

        reducer = TSNE(**tsne_args)
        coords_2d = reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Method must be 'pca' or 'tsne', got '{method}'")

    # === Plotting ===
    fig, ax = plt.subplots(figsize=figsize)

    # Create color palette (supports up to 20 categories)
    cmap = plt.cm.get_cmap('tab20', len(categories))
    colors = [cmap(i) for i in range(len(categories))]

    # Plot each semantic category
    for (cat_name, indices), color in zip(category_indices.items(), colors):
        if not indices:
            continue
        ax.scatter(coords_2d[indices, 0], coords_2d[indices, 1],
                   alpha=0.7, s=100, c=[color], label=cat_name)
        for idx in indices:
            ax.annotate(words[idx],
                        (coords_2d[idx, 0], coords_2d[idx, 1]),
                        fontsize=9,
                        xytext=(4, 4),
                        textcoords='offset points')

    # === Styling ===
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title('Semantic Clustering of Word Embeddings',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Optionally save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_analogy_visualization(
    word_a: str, word_b: str, word_c: str, word_d: str,
    embeddings_dict: Dict[str, np.ndarray],
    method: str = 'pca',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize word analogy: word_a is to word_b as word_c is to word_d.
    Shows the vector relationships in 2D space.
    
    Args:
        word_a, word_b, word_c, word_d: Words in the analogy
        embeddings_dict: Dictionary mapping words to their embedding vectors
        method: 'pca' or 'tsne' for dimensionality reduction
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    # Get embeddings
    words = [word_a, word_b, word_c, word_d]
    embeddings = np.array([embeddings_dict[w] for w in words])
    
    # Dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
    else:
        reducer = TSNE(n_components=2, perplexity=2, n_iter=1000, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    colors = ['blue', 'green', 'blue', 'green']
    sizes = [150, 150, 150, 150]
    
    for i, (word, color, size) in enumerate(zip(words, colors, sizes)):
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], 
                  c=color, s=size, alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(word, (coords_2d[i, 0], coords_2d[i, 1]),
                   fontsize=12, fontweight='bold',
                   xytext=(8, 8), textcoords='offset points')
    
    # Draw arrows showing relationships
    # Arrow from a to b
    ax.annotate('', xy=(coords_2d[1, 0], coords_2d[1, 1]),
               xytext=(coords_2d[0, 0], coords_2d[0, 1]),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.6))
    
    # Arrow from c to d
    ax.annotate('', xy=(coords_2d[3, 0], coords_2d[3, 1]),
               xytext=(coords_2d[2, 0], coords_2d[2, 1]),
               arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.6))
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=11)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=11)
    ax.set_title(f'Word Analogy: "{word_a}" is to "{word_b}" as "{word_c}" is to "{word_d}"',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# ATTENTION VISUALIZATION FUNCTIONS
# ============================================================================

def plot_attention_weights(
    attention_weights: np.ndarray,
    source_tokens: List[str],
    target_tokens: List[str],
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'YlOrRd',
    title: str = 'Attention Weights',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Array of shape (target_len, source_len)
        source_tokens: List of source sequence tokens
        target_tokens: List of target sequence tokens
        figsize: Figure size
        cmap: Colormap name
        title: Plot title
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(source_tokens)))
    ax.set_yticks(np.arange(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha='right')
    ax.set_yticklabels(target_tokens)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(target_tokens)):
        for j in range(len(source_tokens)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                         ha='center', va='center', color='black', fontsize=8)
    
    ax.set_xlabel('Source Tokens', fontsize=11)
    ax.set_ylabel('Target Tokens', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multi_head_attention(
    attention_weights_list: List[np.ndarray],
    source_tokens: List[str],
    target_tokens: List[str],
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize multiple attention heads in a grid.
    
    Args:
        attention_weights_list: List of attention weight matrices, one per head
        source_tokens: List of source sequence tokens
        target_tokens: List of target sequence tokens
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    n_heads = len(attention_weights_list)
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_heads > 1 else [axes]
    
    for idx, (ax, attn) in enumerate(zip(axes, attention_weights_list)):
        im = ax.imshow(attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(source_tokens)))
        ax.set_yticks(np.arange(len(target_tokens)))
        ax.set_xticklabels(source_tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(target_tokens, fontsize=8)
        
        ax.set_title(f'Head {idx + 1}', fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Multi-Head Attention Weights', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# METRIC VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation losses and metrics over epochs.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Dict of metric names to lists of training metric values
        val_metrics: Dict of metric names to lists of validation metric values
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    n_plots = 1 + (1 if train_metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics
    if train_metrics and n_plots > 1:
        for metric_name, values in train_metrics.items():
            axes[1].plot(epochs, values, '-o', label=f'Train {metric_name}', linewidth=2)
        
        if val_metrics:
            for metric_name, values in val_metrics.items():
                axes[1].plot(epochs, values, '-s', label=f'Val {metric_name}', linewidth=2)
        
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Metric Value', fontsize=11)
        axes[1].set_title('Training and Validation Metrics', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: List[str],
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'coolwarm',
    title: str = 'Similarity Matrix',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a similarity/distance matrix as a heatmap.
    
    Args:
        similarity_matrix: Square matrix of similarities/distances
        labels: Labels for rows and columns
        figsize: Figure size
        cmap: Colormap name
        title: Plot title
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if abs(similarity_matrix[i, j]) > 0.5 else 'black',
                         fontsize=8)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_word_similarity_comparison(
    words: List[str],
    similarities_dict: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare similarity scores from different embedding models.
    
    Args:
        words: List of words being compared
        similarities_dict: Dict mapping model names to lists of similarity scores
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(words))
    width = 0.8 / len(similarities_dict)
    
    for idx, (model_name, scores) in enumerate(similarities_dict.items()):
        offset = (idx - len(similarities_dict) / 2) * width + width / 2
        ax.bar(x + offset, scores, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Word Pairs', fontsize=11)
    ax.set_ylabel('Similarity Score', fontsize=11)
    ax.set_title('Word Similarity Comparison Across Models', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig